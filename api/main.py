import json
import joblib
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import RecommendationResponse, RecommendationItem, MetricsResponse, AllMetricsResponse, HealthResponse
from config import MODELS_DIR


MODELS: dict = {}
METRICS: dict = {}


def _load_artifacts():
    for model_name in ["popularity", "user_cf", "item_cf", "svd"]:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if model_path.exists():
            MODELS[model_name] = joblib.load(model_path)
            print(f"Loaded model: {model_name}")

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            METRICS.update(json.load(f))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="Movie Recommendation API",
    description="Personalized movie recommendations powered by Collaborative Filtering and SVD",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(status="ok", models_loaded=list(MODELS.keys()))


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(
    user_id: int,
    model: Optional[str] = Query(default="svd", description="Model: popularity | user_cf | item_cf | svd"),
    n: int = Query(default=10, ge=1, le=50, description="Number of recommendations"),
    exclude_seen: bool = Query(default=True, description="Exclude movies the user has already rated"),
):
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded. Run scripts/train.py first.")

    if model not in MODELS:
        available = list(MODELS.keys())
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found. Available: {available}")

    recommender = MODELS[model]
    try:
        recs = recommender.recommend(user_id, n=n, exclude_seen=exclude_seen)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not recs:
        raise HTTPException(status_code=404, detail=f"No recommendations found for user {user_id}. User may not exist in training data.")

    items = [RecommendationItem(**r) for r in recs]
    return RecommendationResponse(userId=user_id, model=model, recommendations=items)


@app.get("/metrics", response_model=AllMetricsResponse, tags=["Evaluation"])
def get_metrics():
    if not METRICS:
        raise HTTPException(status_code=404, detail="No metrics found. Run scripts/train.py to train and evaluate models.")

    results = []
    for model_name, m in METRICS.items():
        results.append(MetricsResponse(
            model=model_name,
            rmse=m.get("rmse"),
            precision_at_10=m.get("precision_at_10"),
            recall_at_10=m.get("recall_at_10"),
            ndcg_at_10=m.get("ndcg_at_10"),
        ))
    return AllMetricsResponse(metrics=results)


@app.get("/metrics/{model_name}", response_model=MetricsResponse, tags=["Evaluation"])
def get_model_metrics(model_name: str):
    if model_name not in METRICS:
        raise HTTPException(status_code=404, detail=f"Metrics for model '{model_name}' not found.")
    m = METRICS[model_name]
    return MetricsResponse(
        model=model_name,
        rmse=m.get("rmse"),
        precision_at_10=m.get("precision_at_10"),
        recall_at_10=m.get("recall_at_10"),
        ndcg_at_10=m.get("ndcg_at_10"),
    )


@app.get("/models", tags=["System"])
def list_models():
    return {"available_models": list(MODELS.keys())}
