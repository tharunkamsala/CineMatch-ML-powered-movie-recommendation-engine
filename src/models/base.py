from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd


class BaseRecommender(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train_ratings: pd.DataFrame, movies: pd.DataFrame, **kwargs) -> None:
        ...

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> List[Dict[str, Any]]:
        """Return list of dicts with keys: movieId, title, score, explanation."""
        ...

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a single user-movie pair. Override in subclasses."""
        raise NotImplementedError

    def _get_seen(self, user_id: int) -> set:
        if hasattr(self, "_user_seen"):
            return self._user_seen.get(user_id, set())
        return set()

    def _build_seen_index(self, ratings: pd.DataFrame) -> None:
        self._user_seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()
