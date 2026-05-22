"""
End-to-end training pipeline.

Usage:
    python scripts/train.py                          # download small dataset automatically
    python scripts/train.py --dataset 25m            # download MovieLens 25M
    python scripts/train.py --data-path data/raw/extracted/ml-synthetic  # use local folder
    python scripts/train.py --skip-eval              # skip evaluation (faster)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    MOVIELENS_25M_URL,
    MOVIELENS_SMALL_URL,
    SVD_N_COMPONENTS,
    CF_N_NEIGHBORS,
    TOP_N,
)
from src.data.loader import download_dataset, load_ratings, load_movies
from src.data.processor import (
    filter_sparse_users_movies,
    split_data,
    build_index_maps,
    compute_movie_features,
    encode_genres,
)
from src.models.popularity import PopularityRecommender
from src.models.user_cf import UserCFRecommender
from src.models.item_cf import ItemCFRecommender
from src.models.svd_model import SVDRecommender
from src.evaluation.metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["small", "25m"], default="small")
    parser.add_argument("--data-path", type=str, default=None, help="Path to already-extracted MovieLens folder (skips download)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    parser.add_argument("--models", nargs="+", default=["popularity", "user_cf", "item_cf", "svd"],
                        help="Which models to train")
    return parser.parse_args()


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    url = MOVIELENS_25M_URL if args.dataset == "25m" else MOVIELENS_SMALL_URL
    dataset_label = args.data_path if args.data_path else args.dataset.upper()
    print(f"\n{'='*60}")
    print(f"  MovieLens Recommendation System — Training Pipeline")
    print(f"  Dataset: {dataset_label}")
    print(f"{'='*60}\n")

    # ── 1. Download & Load ────────────────────────────────────────
    print("[ 1/5 ] Loading data...")
    if args.data_path:
        data_path = Path(args.data_path)
        print(f"Using local data from: {data_path}")
    else:
        data_path = download_dataset(url)
    ratings = load_ratings(data_path)
    movies = load_movies(data_path)
    movies = encode_genres(movies)

    # ── 2. Preprocess ─────────────────────────────────────────────
    print("\n[ 2/5 ] Preprocessing...")
    ratings = filter_sparse_users_movies(ratings)
    train, test = split_data(ratings)

    # Save processed data for the dashboard
    ratings.to_parquet(PROCESSED_DATA_DIR / "ratings.parquet", index=False)
    movies.to_parquet(PROCESSED_DATA_DIR / "movies.parquet", index=False)
    train.to_parquet(PROCESSED_DATA_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DATA_DIR / "test.parquet", index=False)
    print("Processed data saved.")

    # ── 3. Build model registry ───────────────────────────────────
    model_registry = {
        "popularity": PopularityRecommender(),
        "user_cf": UserCFRecommender(n_neighbors=CF_N_NEIGHBORS),
        "item_cf": ItemCFRecommender(n_neighbors=CF_N_NEIGHBORS),
        "svd": SVDRecommender(n_components=SVD_N_COMPONENTS),
    }
    selected_models = {k: v for k, v in model_registry.items() if k in args.models}

    # ── 4. Train ──────────────────────────────────────────────────
    print(f"\n[ 3/5 ] Training {len(selected_models)} model(s)...")
    for name, model in selected_models.items():
        print(f"\n  → Training {name}...")
        t0 = time.time()
        model.fit(train, movies)
        elapsed = time.time() - t0
        print(f"  ✓ {name} trained in {elapsed:.1f}s")
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"  ✓ Saved to {MODELS_DIR / f'{name}.joblib'}")

    # ── 5. Evaluate ───────────────────────────────────────────────
    all_metrics = {}
    if not args.skip_eval:
        print(f"\n[ 4/5 ] Evaluating models (k={TOP_N})...")
        for name, model in selected_models.items():
            print(f"\n  → Evaluating {name}...")
            t0 = time.time()
            m = evaluate_model(model, test, train, k=TOP_N)
            elapsed = time.time() - t0
            # Rename keys for the API
            all_metrics[name] = {
                "rmse": m["rmse"],
                "precision_at_10": m.get(f"precision_at_{TOP_N}"),
                "recall_at_10": m.get(f"recall_at_{TOP_N}"),
                "ndcg_at_10": m.get(f"ndcg_at_{TOP_N}"),
            }
            print(f"  ✓ {name}: RMSE={m['rmse']:.4f} | "
                  f"P@{TOP_N}={m.get(f'precision_at_{TOP_N}', 'N/A')} | "
                  f"R@{TOP_N}={m.get(f'recall_at_{TOP_N}', 'N/A')} | "
                  f"NDCG@{TOP_N}={m.get(f'ndcg_at_{TOP_N}', 'N/A')} "
                  f"({elapsed:.1f}s)")
    else:
        print("\n[ 4/5 ] Skipping evaluation (--skip-eval).")
        for name in selected_models:
            all_metrics[name] = {"rmse": None, "precision_at_10": None, "recall_at_10": None, "ndcg_at_10": None}

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  ✓ Metrics saved to {metrics_path}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n[ 5/5 ] Done!\n")
    print("  To start the API:")
    print("    uvicorn api.main:app --reload\n")
    print("  To launch the dashboard:")
    print("    streamlit run dashboard/app.py\n")

    if all_metrics:
        print("  Evaluation summary:")
        print(f"  {'Model':<15} {'RMSE':>8} {'P@10':>8} {'R@10':>8} {'NDCG@10':>10}")
        print("  " + "-" * 52)
        for name, m in all_metrics.items():
            def fmt(v): return f"{v:.4f}" if v is not None else "  N/A "
            print(f"  {name:<15} {fmt(m['rmse']):>8} {fmt(m['precision_at_10']):>8} "
                  f"{fmt(m['recall_at_10']):>8} {fmt(m['ndcg_at_10']):>10}")
    print()


if __name__ == "__main__":
    main()
