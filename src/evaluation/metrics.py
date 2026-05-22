import numpy as np
import pandas as pd
from typing import List, Dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    top_k = recommended[:k]
    hits = sum(1 for m in top_k if m in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    top_k = recommended[:k]
    hits = sum(1 for m in top_k if m in relevant)
    return hits / len(relevant) if len(relevant) > 0 else 0.0


def ndcg_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, m in enumerate(top_k)
        if m in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    model,
    test_ratings: pd.DataFrame,
    train_ratings: pd.DataFrame,
    k: int = 10,
    n_users: int = 500,
    relevance_threshold: float = 4.0,
) -> Dict[str, float]:
    """
    Evaluate a recommender model on RMSE and ranking metrics.

    For ranking metrics, a movie is 'relevant' if the user rated it >= relevance_threshold in test.
    We recommend from movies NOT in the user's training set.
    """
    # --- RMSE ---
    y_true, y_pred = [], []
    test_sample = test_ratings.sample(min(5000, len(test_ratings)), random_state=42)
    for _, row in test_sample.iterrows():
        try:
            pred = model.predict_rating(int(row["userId"]), int(row["movieId"]))
            y_true.append(row["rating"])
            y_pred.append(pred)
        except Exception:
            continue

    rmse_val = rmse(np.array(y_true), np.array(y_pred)) if y_true else float("nan")

    # --- Ranking metrics ---
    test_users = (
        test_ratings[test_ratings["rating"] >= relevance_threshold]["userId"]
        .value_counts()
        .head(n_users)
        .index.tolist()
    )

    precisions, recalls, ndcgs = [], [], []
    for user_id in test_users:
        relevant = set(
            test_ratings[(test_ratings["userId"] == user_id) & (test_ratings["rating"] >= relevance_threshold)]["movieId"]
        )
        if not relevant:
            continue
        try:
            recs = model.recommend(user_id, n=k, exclude_seen=True)
            rec_ids = [r["movieId"] for r in recs]
        except Exception:
            continue

        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))

    return {
        "rmse": round(rmse_val, 4),
        f"precision_at_{k}": round(np.mean(precisions), 4) if precisions else float("nan"),
        f"recall_at_{k}": round(np.mean(recalls), 4) if recalls else float("nan"),
        f"ndcg_at_{k}": round(np.mean(ndcgs), 4) if ndcgs else float("nan"),
    }
