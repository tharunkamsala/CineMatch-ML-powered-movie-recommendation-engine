from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from src.models.base import BaseRecommender
from src.data.processor import build_index_maps, build_user_item_matrix


class UserCFRecommender(BaseRecommender):
    """User-user collaborative filtering using cosine similarity."""
    name = "user_cf"

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors

    def fit(self, train_ratings: pd.DataFrame, movies: pd.DataFrame, **kwargs) -> None:
        self._build_seen_index(train_ratings)
        self._movies = movies.set_index("movieId")

        self._user2idx, self._movie2idx, self._idx2user, self._idx2movie = build_index_maps(train_ratings)
        self._matrix = build_user_item_matrix(train_ratings, self._user2idx, self._movie2idx)

        print(f"Fitting User CF with {self.n_neighbors} neighbors...")
        self._knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=self.n_neighbors + 1, n_jobs=-1)
        self._knn.fit(self._matrix)
        print("User CF fit complete.")

    def _get_user_vector(self, user_id: int):
        if user_id not in self._user2idx:
            return None
        idx = self._user2idx[user_id]
        return self._matrix[idx]

    def _get_neighbor_ids(self, user_id: int):
        vec = self._get_user_vector(user_id)
        if vec is None:
            return [], []
        distances, indices = self._knn.kneighbors(vec, n_neighbors=self.n_neighbors + 1)
        distances, indices = distances[0][1:], indices[0][1:]  # exclude self
        similarities = 1 - distances
        neighbor_ids = [self._idx2user[i] for i in indices]
        return neighbor_ids, similarities

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> List[Dict[str, Any]]:
        seen = self._get_seen(user_id) if exclude_seen else set()
        neighbor_ids, similarities = self._get_neighbor_ids(user_id)

        if not neighbor_ids:
            return []

        # Aggregate neighbor ratings weighted by similarity
        scores: dict[int, list] = {}
        for neighbor_id, sim in zip(neighbor_ids, similarities):
            for movie_id in self._user_seen.get(neighbor_id, set()):
                if movie_id not in seen and movie_id in self._movie2idx:
                    scores.setdefault(movie_id, []).append((sim, self._get_rating(neighbor_id, movie_id)))

        # Weighted average
        ranked = []
        for movie_id, sim_rating_pairs in scores.items():
            total_sim = sum(s for s, _ in sim_rating_pairs)
            if total_sim == 0:
                continue
            pred = sum(s * r for s, r in sim_rating_pairs) / total_sim
            ranked.append((movie_id, pred, len(sim_rating_pairs)))

        ranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for movie_id, score, n_neighbors in ranked[:n]:
            title = self._movies.loc[movie_id, "title"] if movie_id in self._movies.index else "Unknown"
            results.append({
                "movieId": int(movie_id),
                "title": title,
                "score": round(float(score), 4),
                "explanation": f"Liked by {n_neighbors} users with similar taste (predicted rating: {score:.2f}/5)",
            })
        return results

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        neighbor_ids, similarities = self._get_neighbor_ids(user_id)
        if not neighbor_ids:
            return 3.5

        weighted_sum = 0.0
        sim_sum = 0.0
        for neighbor_id, sim in zip(neighbor_ids, similarities):
            if movie_id in self._user_seen.get(neighbor_id, set()):
                r = self._get_rating(neighbor_id, movie_id)
                weighted_sum += sim * r
                sim_sum += sim

        return weighted_sum / sim_sum if sim_sum > 0 else 3.5

    def _get_rating(self, user_id: int, movie_id: int) -> float:
        u_idx = self._user2idx.get(user_id)
        m_idx = self._movie2idx.get(movie_id)
        if u_idx is None or m_idx is None:
            return 3.5
        val = self._matrix[u_idx, m_idx]
        return float(val) if val != 0 else 3.5
