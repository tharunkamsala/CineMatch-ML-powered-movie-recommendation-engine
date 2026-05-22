import json
import joblib
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    RecommendationResponse, RecommendationItem,
    MetricsResponse, AllMetricsResponse, HealthResponse,
)
from config import MODELS_DIR
from src.realtime.cache import RecommendationCache
from src.realtime.stream import RatingStream, RatingEvent
from src.realtime.online_learner import OnlineSVDLearner
from src.ab_testing.bandit import ModelBandit
from src.ab_testing.tracker import CTRTracker
from src.ab_testing.shadow import ShadowDeployment

# ── Global state ──────────────────────────────────────────────────────────────
MODELS:  dict = {}
METRICS: dict = {}
CACHE:   Optional[RecommendationCache] = None
STREAM:  Optional[RatingStream]        = None
LEARNER: Optional[OnlineSVDLearner]    = None
BANDIT:  Optional[ModelBandit]         = None
TRACKER: Optional[CTRTracker]          = None
SHADOW:  Optional[ShadowDeployment]    = None


def _load_artifacts():
    for model_name in ["popularity", "user_cf", "item_cf", "svd", "ncf", "two_tower"]:
        path = MODELS_DIR / f"{model_name}.joblib"
        if path.exists():
            MODELS[model_name] = joblib.load(path)

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            METRICS.update(json.load(f))


def _init_services():
    global CACHE, STREAM, LEARNER, BANDIT, TRACKER, SHADOW

    # Cache (Redis optional — falls back to in-memory)
    CACHE = RecommendationCache(redis_url=None, ttl=3600)

    # Rating stream (Kafka optional — falls back to in-memory queue)
    STREAM = RatingStream(kafka_url=None)

    # Online learner (hooks into SVD model)
    if "svd" in MODELS:
        LEARNER = OnlineSVDLearner(MODELS["svd"], lr=0.01, cache=CACHE)
        STREAM.consume(callback=LEARNER.handle_event, block=False)

    # CTR tracker
    TRACKER = CTRTracker(db_path=MODELS_DIR / "experiments.db")

    # Multi-armed bandit over available models
    BANDIT = ModelBandit(
        model_names=list(MODELS.keys()),
        strategy="ucb",
        state_path=MODELS_DIR / "bandit_state.json",
    )

    # Shadow deployment: SVD as primary, others as shadows
    if "svd" in MODELS:
        shadows = {k: v for k, v in MODELS.items() if k != "svd"}
        SHADOW  = ShadowDeployment(MODELS["svd"], shadows, tracker=TRACKER)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    _init_services()
    print(f"Loaded models: {list(MODELS.keys())}")
    yield


app = FastAPI(
    title="CineMatch — Movie Recommendation API",
    description="Personalized recommendations via CF, SVD, NCF, Two-Tower with A/B testing & real-time updates",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(status="ok", models_loaded=list(MODELS.keys()))


@app.get("/models", tags=["System"])
def list_models():
    return {"available_models": list(MODELS.keys())}


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(
    user_id: int,
    model:        Optional[str] = Query(default=None, description="Model name. Omit to let the bandit choose."),
    n:            int  = Query(default=10, ge=1, le=50),
    exclude_seen: bool = Query(default=True),
    use_cache:    bool = Query(default=True),
):
    if not MODELS:
        raise HTTPException(503, "No models loaded. Run scripts/train.py first.")

    # Bandit selects model if not specified
    chosen_model = model or BANDIT.select(user_id)
    if chosen_model not in MODELS:
        raise HTTPException(404, f"Model '{chosen_model}' not found. Available: {list(MODELS.keys())}")

    # Cache lookup
    cache_key = RecommendationCache.make_key(user_id, chosen_model, n)
    if use_cache and CACHE:
        cached = CACHE.get(cache_key)
        if cached is not None:
            return RecommendationResponse(userId=user_id, model=chosen_model,
                                          recommendations=[RecommendationItem(**r) for r in cached])

    # Generate recommendations
    try:
        recs = MODELS[chosen_model].recommend(user_id, n=n, exclude_seen=exclude_seen)
    except Exception as e:
        raise HTTPException(500, str(e))

    if not recs:
        raise HTTPException(404, f"No recommendations for user {user_id}.")

    # Log impression for CTR tracking
    if TRACKER:
        TRACKER.log_impression(user_id, chosen_model, [r["movieId"] for r in recs])

    # Cache result
    if use_cache and CACHE:
        CACHE.set(cache_key, recs)

    items = [RecommendationItem(**r) for r in recs]
    return RecommendationResponse(userId=user_id, model=chosen_model, recommendations=items)


@app.get("/recommendations/{user_id}/shadow", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations_shadow(
    user_id: int,
    n: int = Query(default=10, ge=1, le=50),
    exclude_seen: bool = Query(default=True),
):
    """Serve from primary model (SVD) while silently running all shadows."""
    if SHADOW is None:
        raise HTTPException(503, "Shadow deployment not initialised.")
    recs = SHADOW.recommend(user_id, n=n, exclude_seen=exclude_seen, log_impression=True)
    if not recs:
        raise HTTPException(404, f"No recommendations for user {user_id}.")
    return RecommendationResponse(userId=user_id, model="svd (shadow)",
                                  recommendations=[RecommendationItem(**r) for r in recs])


# ── Feedback / online learning ────────────────────────────────────────────────

@app.post("/feedback", tags=["Feedback"])
def submit_feedback(
    user_id:  int,
    movie_id: int,
    rating:   float = Query(ge=0.5, le=5.0),
    model:    str   = Query(default="svd"),
    background_tasks: BackgroundTasks = None,
):
    """
    Submit a new rating. Triggers:
    1. Online SVD embedding update
    2. CTR click logged
    3. Bandit reward update
    """
    event = RatingEvent(user_id=user_id, movie_id=movie_id, rating=rating)

    if STREAM:
        STREAM.produce(event)

    if TRACKER:
        reward = 1.0 if rating >= 4.0 else 0.5 if rating >= 3.0 else 0.0
        TRACKER.log_click(user_id, movie_id, model, reward=reward)
        if BANDIT:
            BANDIT.update(model, reward)

    return {"status": "accepted", "user_id": user_id, "movie_id": movie_id, "rating": rating}


# ── Metrics & analytics ───────────────────────────────────────────────────────

@app.get("/metrics", response_model=AllMetricsResponse, tags=["Evaluation"])
def get_metrics():
    if not METRICS:
        raise HTTPException(404, "No metrics. Run scripts/train.py.")
    results = [
        MetricsResponse(
            model=name,
            rmse=m.get("rmse"),
            precision_at_10=m.get("precision_at_10"),
            recall_at_10=m.get("recall_at_10"),
            ndcg_at_10=m.get("ndcg_at_10"),
        )
        for name, m in METRICS.items()
    ]
    return AllMetricsResponse(metrics=results)


@app.get("/metrics/{model_name}", response_model=MetricsResponse, tags=["Evaluation"])
def get_model_metrics(model_name: str):
    if model_name not in METRICS:
        raise HTTPException(404, f"Metrics for '{model_name}' not found.")
    m = METRICS[model_name]
    return MetricsResponse(model=model_name, rmse=m.get("rmse"),
                           precision_at_10=m.get("precision_at_10"),
                           recall_at_10=m.get("recall_at_10"),
                           ndcg_at_10=m.get("ndcg_at_10"))


@app.get("/ab/bandit", tags=["A/B Testing"])
def bandit_stats():
    if BANDIT is None:
        raise HTTPException(503, "Bandit not initialised.")
    return BANDIT.stats()


@app.get("/ab/shadow", tags=["A/B Testing"])
def shadow_report():
    if SHADOW is None:
        raise HTTPException(503, "Shadow deployment not initialised.")
    return SHADOW.shadow_report()


@app.get("/ab/ctr", tags=["A/B Testing"])
def ctr_report(model: Optional[str] = None):
    if TRACKER is None:
        raise HTTPException(503, "CTR tracker not initialised.")
    return TRACKER.summary() if model is None else TRACKER.ctr(model)


@app.get("/cache/stats", tags=["System"])
def cache_stats():
    if CACHE is None:
        raise HTTPException(503, "Cache not initialised.")
    return CACHE.stats()


@app.get("/stream/stats", tags=["System"])
def stream_stats():
    if STREAM is None:
        raise HTTPException(503, "Stream not initialised.")
    return STREAM.stats()
