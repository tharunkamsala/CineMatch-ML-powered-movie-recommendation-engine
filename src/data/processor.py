import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MIN_RATINGS_PER_USER, MIN_RATINGS_PER_MOVIE, TEST_SIZE, RANDOM_STATE


def filter_sparse_users_movies(
    ratings: pd.DataFrame,
    min_user_ratings: int = MIN_RATINGS_PER_USER,
    min_movie_ratings: int = MIN_RATINGS_PER_MOVIE,
) -> pd.DataFrame:
    """Iteratively filter users and movies with too few ratings."""
    prev_len = -1
    while prev_len != len(ratings):
        prev_len = len(ratings)
        user_counts = ratings["userId"].value_counts()
        ratings = ratings[ratings["userId"].isin(user_counts[user_counts >= min_user_ratings].index)]
        movie_counts = ratings["movieId"].value_counts()
        ratings = ratings[ratings["movieId"].isin(movie_counts[movie_counts >= min_movie_ratings].index)]
    print(f"After filtering: {len(ratings):,} ratings, {ratings['userId'].nunique():,} users, {ratings['movieId'].nunique():,} movies")
    return ratings.reset_index(drop=True)


def split_data(ratings: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    print(f"Train: {len(train):,} | Test: {len(test):,}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_index_maps(ratings: pd.DataFrame):
    """Build contiguous integer indices for users and movies."""
    users = sorted(ratings["userId"].unique())
    movies = sorted(ratings["movieId"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    movie2idx = {m: i for i, m in enumerate(movies)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2movie = {i: m for m, i in movie2idx.items()}
    return user2idx, movie2idx, idx2user, idx2movie


def build_user_item_matrix(ratings: pd.DataFrame, user2idx: dict, movie2idx: dict) -> csr_matrix:
    """Build a sparse user-item rating matrix."""
    rows = ratings["userId"].map(user2idx)
    cols = ratings["movieId"].map(movie2idx)
    data = ratings["rating"].values.astype(np.float32)

    mask = rows.notna() & cols.notna()
    rows, cols, data = rows[mask].astype(int), cols[mask].astype(int), data[mask]

    shape = (len(user2idx), len(movie2idx))
    matrix = csr_matrix((data, (rows, cols)), shape=shape)
    print(f"User-item matrix shape: {matrix.shape}, density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")
    return matrix


def compute_movie_features(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Compute per-movie aggregate features."""
    stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        rating_std=("rating", "std"),
    ).reset_index()
    stats["rating_std"] = stats["rating_std"].fillna(0)

    # Bayesian average (smoothed by global mean) to penalise movies with few ratings
    global_mean = ratings["rating"].mean()
    m = 10  # minimum count weight
    stats["bayes_avg"] = (stats["rating_count"] * stats["avg_rating"] + m * global_mean) / (stats["rating_count"] + m)

    stats = stats.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    return stats


def encode_genres(movies: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode pipe-separated genre strings."""
    all_genres = set()
    movies["genres"].dropna().str.split("|").apply(all_genres.update)
    all_genres.discard("(no genres listed)")

    for genre in sorted(all_genres):
        movies[f"genre_{genre}"] = movies["genres"].str.contains(genre, regex=False).astype(int)
    return movies
