from typing import List, Dict, Any
import pandas as pd

from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def fit(self, train_ratings: pd.DataFrame, movies: pd.DataFrame, **kwargs) -> None:
        self._build_seen_index(train_ratings)

        global_mean = train_ratings["rating"].mean()
        m = 50  # Bayesian smoothing weight

        stats = train_ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            count=("rating", "count"),
        ).reset_index()
        stats["score"] = (stats["count"] * stats["avg_rating"] + m * global_mean) / (stats["count"] + m)
        stats = stats.merge(movies[["movieId", "title"]], on="movieId", how="left")
        stats = stats.sort_values("score", ascending=False).reset_index(drop=True)
        stats["rank"] = stats.index + 1

        self._ranking = stats
        self._total = len(stats)

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> List[Dict[str, Any]]:
        seen = self._get_seen(user_id) if exclude_seen else set()
        candidates = self._ranking[~self._ranking["movieId"].isin(seen)]

        results = []
        for _, row in candidates.head(n).iterrows():
            pct = round((1 - row["rank"] / self._total) * 100, 1)
            results.append({
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "score": round(float(row["score"]), 4),
                "explanation": f"Top {pct}% most popular — rated by {int(row['count']):,} users with avg {row['avg_rating']:.2f}/5",
            })
        return results

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        row = self._ranking[self._ranking["movieId"] == movie_id]
        if row.empty:
            return self._ranking["score"].mean()
        return float(row.iloc[0]["score"])
