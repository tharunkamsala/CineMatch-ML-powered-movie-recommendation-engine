"""
Online learning: update SVD user embeddings incrementally when new ratings arrive.
No full retraining — only the affected user's latent vector is updated via SGD.
"""
import numpy as np
from typing import Optional
from src.realtime.stream import RatingEvent


class OnlineSVDLearner:
    """
    Wraps a trained SVDRecommender and applies mini-batch SGD updates
    to the user embedding whenever a new rating event is received.

    Math:
        loss = (rating - user_vec · item_vec - user_bias)²
        user_vec  -= lr * d_loss/d_user_vec
        user_bias -= lr * d_loss/d_user_bias
    """

    def __init__(self, svd_model, lr: float = 0.01, cache=None):
        self.model = svd_model
        self.lr    = lr
        self.cache = cache
        self._updates = 0

    def handle_event(self, event: RatingEvent) -> None:
        """Process one incoming rating event and update the user embedding."""
        user_id  = event.user_id
        movie_id = event.movie_id
        rating   = np.clip(event.rating, 0.5, 5.0)

        u_idx = self.model._user2idx.get(user_id)
        m_idx = self.model._movie2idx.get(movie_id)

        if u_idx is None or m_idx is None:
            # Unknown user or movie — skip (cold-start handled separately)
            return

        # Current user embedding from U matrix
        # SVD stores: _U (n_users, k), _sigma (k,), _Vt (k, n_items), _user_means
        sigma_vt = np.diag(self.model._sigma) @ self.model._Vt  # (k, n_items)
        item_vec  = sigma_vt[:, m_idx]                           # (k,)
        user_vec  = self.model._U[u_idx].copy()                  # (k,)
        user_bias = self.model._user_means[u_idx]

        pred  = float(np.dot(user_vec, item_vec) + user_bias)
        error = rating - pred

        # Gradient descent step on user vector only (item embeddings unchanged)
        grad_user = -2 * error * item_vec
        self.model._U[u_idx] -= self.lr * grad_user

        # Update predicted matrix row for this user
        self.model._predicted[u_idx] = (
            self.model._U[u_idx] @ sigma_vt + user_bias
        )
        np.clip(self.model._predicted[u_idx], 0.5, 5.0, out=self.model._predicted[u_idx])

        # Invalidate cached recommendations for this user
        if self.cache is not None:
            self.cache.invalidate_user(user_id)

        self._updates += 1

    def stats(self) -> dict:
        return {"total_updates": self._updates, "learning_rate": self.lr}
