from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from src.models.base import BaseRecommender
from src.data.processor import build_index_maps, build_user_item_matrix


class SVDRecommender(BaseRecommender):
    """
    Matrix factorization via Truncated SVD.
    Centers ratings by user mean before decomposition so cold predictions
    degrade gracefully to the user's average rating.
    """
    name = "svd"

    def __init__(self, n_components: int = 100):
        self.n_components = n_components

    def fit(self, train_ratings: pd.DataFrame, movies: pd.DataFrame, **kwargs) -> None:
        self._build_seen_index(train_ratings)
        self._movies = movies.set_index("movieId")

        self._user2idx, self._movie2idx, self._idx2user, self._idx2movie = build_index_maps(train_ratings)
        matrix = build_user_item_matrix(train_ratings, self._user2idx, self._movie2idx).astype(np.float64)

        # Center each user's ratings by their mean
        self._user_means = np.zeros(matrix.shape[0])
        cx = matrix.tocsr()
        for u in range(cx.shape[0]):
            row = cx.getrow(u)
            nz = row.data
            if len(nz) > 0:
                self._user_means[u] = nz.mean()

        # Subtract user means from non-zero entries
        centered = cx.copy().astype(np.float64)
        for u in range(centered.shape[0]):
            row_start = centered.indptr[u]
            row_end = centered.indptr[u + 1]
            centered.data[row_start:row_end] -= self._user_means[u]

        k = min(self.n_components, min(matrix.shape) - 1)
        print(f"Running SVD with k={k} components on {matrix.shape} matrix...")
        self._U, self._sigma, self._Vt = svds(centered, k=k)

        # Sort by descending singular value (svds returns in ascending order)
        idx = np.argsort(self._sigma)[::-1]
        self._U = self._U[:, idx]
        self._sigma = self._sigma[idx]
        self._Vt = self._Vt[idx, :]

        # Precompute full prediction matrix (may be large for 25M; use on-demand for big datasets)
        self._predicted = self._U @ np.diag(self._sigma) @ self._Vt
        # Add user means back
        self._predicted += self._user_means[:, np.newaxis]
        print("SVD fit complete.")

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> List[Dict[str, Any]]:
        if user_id not in self._user2idx:
            return []

        seen = self._get_seen(user_id) if exclude_seen else set()
        u_idx = self._user2idx[user_id]
        scores = self._predicted[u_idx].copy()

        # Zero out seen movies
        for movie_id in seen:
            if movie_id in self._movie2idx:
                scores[self._movie2idx[movie_id]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        results = []
        for m_idx in top_indices:
            movie_id = self._idx2movie[m_idx]
            score = float(scores[m_idx])
            if score == -np.inf:
                break
            title = self._movies.loc[movie_id, "title"] if movie_id in self._movies.index else "Unknown"
            results.append({
                "movieId": int(movie_id),
                "title": title,
                "score": round(np.clip(score, 0.5, 5.0).item(), 4),
                "explanation": f"Predicted {score:.2f}/5 based on your latent preference profile (SVD)",
            })
        return results

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        u_idx = self._user2idx.get(user_id)
        m_idx = self._movie2idx.get(movie_id)
        if u_idx is None or m_idx is None:
            return 3.5
        return float(np.clip(self._predicted[u_idx, m_idx], 0.5, 5.0))
