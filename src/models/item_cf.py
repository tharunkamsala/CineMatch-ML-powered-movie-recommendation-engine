from typing import List, Dict, Any
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ITEM_CF_MAX_SEEDS
from src.models.base import BaseRecommender
from src.data.processor import build_index_maps, build_user_item_matrix


class ItemCFRecommender(BaseRecommender):
    """Item-item collaborative filtering using cosine similarity."""
    name = "item_cf"

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors

    def fit(self, train_ratings: pd.DataFrame, movies: pd.DataFrame, **kwargs) -> None:
        self._build_seen_index(train_ratings)
        self._movies = movies.set_index("movieId")

        self._user2idx, self._movie2idx, self._idx2user, self._idx2movie = build_index_maps(train_ratings)
        user_item = build_user_item_matrix(train_ratings, self._user2idx, self._movie2idx)
        self._item_user = user_item.T.tocsr()  # shape: (movies, users)

        print(f"Fitting Item CF with {self.n_neighbors} neighbors...")
        self._knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=self.n_neighbors + 1, n_jobs=-1)
        self._knn.fit(self._item_user)
        print("Item CF fit complete.")

        self._user_item = user_item  # keep for rating lookups

    def _get_similar_items(self, movie_id: int):
        if movie_id not in self._movie2idx:
            return [], []
        m_idx = self._movie2idx[movie_id]
        distances, indices = self._knn.kneighbors(self._item_user[m_idx], n_neighbors=self.n_neighbors + 1)
        distances, indices = distances[0][1:], indices[0][1:]
        similarities = 1 - distances
        similar_ids = [self._idx2movie[i] for i in indices]
        return similar_ids, similarities

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> List[Dict[str, Any]]:
        seen = self._get_seen(user_id) if exclude_seen else set()
        if user_id not in self._user2idx:
            return []

        # Score each unseen movie based on similarity to user's rated movies
        scores: dict[int, list] = {}
        source_movies: dict[int, list] = {}

        u_idx = self._user2idx[user_id]

        # Use only the user's top-rated seed movies to keep recommend() O(SEEDS × n_neighbors)
        rated_with_score = []
        for rated_movie in seen:
            if rated_movie not in self._movie2idx:
                continue
            m_idx = self._movie2idx[rated_movie]
            user_rating = float(self._user_item[u_idx, m_idx])
            if user_rating > 0:
                rated_with_score.append((rated_movie, user_rating))

        rated_with_score.sort(key=lambda x: x[1], reverse=True)
        seed_movies = rated_with_score[:ITEM_CF_MAX_SEEDS]

        for rated_movie, user_rating in seed_movies:
            similar_ids, similarities = self._get_similar_items(rated_movie)
            for sim_movie, sim in zip(similar_ids, similarities):
                if sim_movie not in seen:
                    scores.setdefault(sim_movie, []).append(sim * user_rating)
                    source_movies.setdefault(sim_movie, []).append(
                        self._movies.loc[rated_movie, "title"] if rated_movie in self._movies.index else str(rated_movie)
                    )

        ranked = [(mid, np.mean(s), source_movies[mid]) for mid, s in scores.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for movie_id, score, sources in ranked[:n]:
            title = self._movies.loc[movie_id, "title"] if movie_id in self._movies.index else "Unknown"
            top_sources = list(dict.fromkeys(sources))[:3]  # deduplicate, keep order
            explanation = f"Because you enjoyed: {', '.join(top_sources)}"
            results.append({
                "movieId": int(movie_id),
                "title": title,
                "score": round(float(score), 4),
                "explanation": explanation,
            })
        return results

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if user_id not in self._user2idx or movie_id not in self._movie2idx:
            return 3.5

        similar_ids, similarities = self._get_similar_items(movie_id)
        u_idx = self._user2idx[user_id]

        weighted_sum = 0.0
        sim_sum = 0.0
        for sim_movie, sim in zip(similar_ids, similarities):
            if sim_movie in self._movie2idx:
                m_idx = self._movie2idx[sim_movie]
                rating = float(self._user_item[u_idx, m_idx])
                if rating != 0:
                    weighted_sum += sim * rating
                    sim_sum += sim

        return weighted_sum / sim_sum if sim_sum > 0 else 3.5
