from pydantic import BaseModel
from typing import List, Optional


class RecommendationItem(BaseModel):
    movieId: int
    title: str
    score: float
    explanation: Optional[str] = None


class RecommendationResponse(BaseModel):
    userId: int
    model: str
    recommendations: List[RecommendationItem]


class MetricsResponse(BaseModel):
    model: str
    rmse: Optional[float]
    precision_at_10: Optional[float]
    recall_at_10: Optional[float]
    ndcg_at_10: Optional[float]


class AllMetricsResponse(BaseModel):
    metrics: List[MetricsResponse]


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
