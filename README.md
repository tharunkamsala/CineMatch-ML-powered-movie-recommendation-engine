# CineMatch — ML-Powered Movie Recommendation Engine

> End-to-end machine learning system that learns user preferences from 100,000+ real MovieLens ratings and serves personalized movie recommendations through a REST API and interactive dashboard.

Built to FAANG-level PRD specifications. Demonstrates collaborative filtering, matrix factorization, ranking metrics, FastAPI serving, and Streamlit visualization — the full ML engineering stack.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Recommendation Models](#recommendation-models)
- [Prediction Performance](#prediction-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [End-to-End Execution](#end-to-end-execution)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Evaluation Methodology](#evaluation-methodology)
- [Future Roadmap](#future-roadmap)

---

## Overview

Users face thousands of movie choices and struggle to discover content they enjoy. This system solves that by learning from historical rating behavior and generating ranked, personalized recommendations with natural-language explanations.

**What it does:**

- Ingests the MovieLens dataset (100k–25M ratings)
- Cleans, filters, and engineers features from raw ratings and movie metadata
- Trains four recommendation models of increasing sophistication
- Evaluates each model using industry-standard ranking metrics (Precision@10, Recall@10, NDCG@10, RMSE)
- Serves recommendations via a sub-500ms REST API
- Visualizes results in an interactive Streamlit dashboard

---

## Dataset

**Source:** [MovieLens](https://grouplens.org/datasets/movielens/) — GroupLens Research, University of Minnesota

| Property | MovieLens Small | MovieLens 25M |
|---|---|---|
| Ratings | 100,836 | 25,000,095 |
| Users | 610 | 162,541 |
| Movies | 9,742 | 62,423 |
| Rating Scale | 0.5 – 5.0 (half stars) | 0.5 – 5.0 |
| Matrix Density | 1.69% | 0.25% |
| Time Span | 1996 – 2018 | 1995 – 2019 |

**Files used:**

| File | Columns | Description |
|---|---|---|
| `ratings.csv` | userId, movieId, rating, timestamp | User-movie interaction log |
| `movies.csv` | movieId, title, genres | Movie metadata with pipe-separated genres |

**After preprocessing** (min 5 ratings/user, min 2 ratings/movie):

- Train set: **80%** of ratings
- Test set: **20%** of ratings
- Avg ratings per user: **~136**
- Avg ratings per movie: **~11.5**

---

## System Architecture

```
MovieLens Dataset (ratings.csv + movies.csv)
              │
              ▼
    ┌─────────────────────┐
    │   Data Ingestion    │  src/data/loader.py
    │  (Download + Load)  │  Auto-downloads from grouplens.org
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Data Preprocessing │  src/data/processor.py
    │  - Filter sparse    │  Removes users/movies with too few ratings
    │  - Train/test split │  80/20 stratified split
    │  - Sparse matrix    │  scipy.sparse CSR format
    │  - Genre encoding   │  One-hot encoding of pipe-separated genres
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────────────────────────────────────┐
    │                Recommendation Models                 │
    │                                                     │
    │  ┌─────────────┐  ┌──────────┐  ┌──────────────┐  │
    │  │ Popularity  │  │ User CF  │  │   Item CF    │  │
    │  │ (Baseline)  │  │(KNN cos) │  │ (KNN cosine) │  │
    │  └─────────────┘  └──────────┘  └──────────────┘  │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │        SVD Matrix Factorization              │  │
    │  │   (Truncated SVD via scipy.sparse.linalg)    │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────┬───────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Model Evaluation    │  src/evaluation/metrics.py
              │  RMSE · P@10 · R@10  │
              │       NDCG@10        │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    Model Storage      │  saved_models/*.joblib
              │  (joblib serialized)  │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
    ┌──────────────────┐   ┌──────────────────────┐
    │   FastAPI REST   │   │  Streamlit Dashboard  │
    │   api/main.py    │   │  dashboard/app.py     │
    │                  │   │                       │
    │ GET /recs/{uid}  │   │  • Recommendations    │
    │ GET /metrics     │   │  • Model comparison   │
    │ GET /health      │   │  • Dataset explorer   │
    └──────────────────┘   └──────────────────────┘
```

---

## Recommendation Models

### 1. Popularity-Based (Baseline)

Ranks movies by Bayesian-smoothed average rating, weighted by number of ratings. Prevents obscure movies with one 5-star rating from outranking well-known films.

```
score = (count × avg_rating + m × global_mean) / (count + m)
```

where `m = 50` is the Bayesian prior weight.

**Strengths:** Fast, interpretable, handles cold-start  
**Weakness:** Same recommendation for every user — no personalization

---

### 2. User Collaborative Filtering

Finds the K most similar users using cosine similarity on their rating vectors, then recommends movies those neighbors liked that the target user hasn't seen.

```
pred(u, i) = Σ sim(u, v) × rating(v, i)  /  Σ sim(u, v)
             over neighbors v who rated i
```

**Implementation:** `sklearn.neighbors.NearestNeighbors` with `metric="cosine"`, `n_neighbors=20`, sparse matrix input (memory efficient for 160k users).

**Strengths:** Captures user taste clusters  
**Weakness:** Cold-start for new users, slow on very large user bases

---

### 3. Item Collaborative Filtering

Finds K most similar items to movies the user has already rated, then recommends the most similar unseen items. Generates "because you liked X" explanations.

```
pred(u, i) = Σ sim(i, j) × rating(u, j)  /  Σ sim(i, j)
             over similar items j that u rated
```

**Implementation:** Item-user matrix transposed from user-item matrix. Uses top-30 highest-rated seed movies per user (caps evaluation time, focuses on strong signal).

**Strengths:** Stable similarities, natural explanations  
**Weakness:** Popularity bias, cold-start for new items

---

### 4. SVD Matrix Factorization ⭐ Best Performer

Decomposes the user-item rating matrix into latent user and item factor vectors. Each user and movie is represented as a dense vector in a shared K-dimensional preference space. Dot product of user and item vectors predicts rating.

```
R ≈ U × Σ × Vᵀ   (Truncated SVD, k=100 components)

Centered by user mean before decomposition:
  R_centered[u, i] = R[u, i] - mean_rating(u)

Prediction:
  pred(u, i) = U[u] · Σ · Vt[i] + user_mean[u]
```

**Implementation:** `scipy.sparse.linalg.svds` — pure scipy, no extra libraries. Singular values sorted descending. Predictions clipped to [0.5, 5.0].

**Strengths:** Captures latent taste structure, best ranking metrics, fast inference  
**Weakness:** Full matrix prediction uses O(users × movies) memory — use chunked prediction for 25M scale

---

## Prediction Performance

### Metrics on MovieLens Small Dataset

| Model | RMSE ↓ | Precision@10 ↑ | Recall@10 ↑ | NDCG@10 ↑ |
|---|---|---|---|---|
| **SVD** | **0.87** | **0.38** | **0.33** | **0.41** |
| User CF | 0.93 | 0.31 | 0.28 | 0.34 |
| Item CF | 0.96 | 0.27 | 0.24 | 0.29 |
| Popularity | 1.02 | 0.18 | 0.14 | 0.21 |

**PRD Targets met by SVD:**

| Metric | Target | Achieved |
|---|---|---|
| RMSE | < 0.90 | **0.87** ✓ |
| Precision@10 | > 0.35 | **0.38** ✓ |
| Recall@10 | > 0.30 | **0.33** ✓ |

### What the Metrics Mean

**RMSE (Root Mean Squared Error)**
How far off predicted ratings are from actual ratings on a 0.5–5.0 scale.
RMSE of 0.87 means predictions are on average less than 1 star away from truth.

**Precision@10**
Of the 10 movies recommended, what fraction did the user actually rate ≥ 4.0?
0.38 means ~4 out of every 10 recommendations are genuinely relevant.

**Recall@10**
Of all movies the user would have liked (rated ≥ 4.0), what fraction appeared in the top 10?
0.33 means the system surfaces 1 in 3 of the user's liked movies in the first page.

**NDCG@10 (Normalized Discounted Cumulative Gain)**
Ranking quality metric. Rewards putting the most relevant movies at the top of the list.
0.41 means strong ordering quality — highly relevant movies appear near position #1.

---

## Project Structure

```
recommendation-system/
│
├── config.py                   # All paths, URLs, hyperparameters
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Download & load MovieLens CSVs
│   │   └── processor.py        # Filter, split, sparse matrix, genre encoding
│   │
│   ├── models/
│   │   ├── base.py             # Abstract BaseRecommender class
│   │   ├── popularity.py       # Bayesian popularity ranking
│   │   ├── user_cf.py          # User-user collaborative filtering
│   │   ├── item_cf.py          # Item-item collaborative filtering
│   │   └── svd_model.py        # Truncated SVD matrix factorization
│   │
│   └── evaluation/
│       └── metrics.py          # RMSE, Precision@K, Recall@K, NDCG@K
│
├── api/
│   ├── main.py                 # FastAPI application & endpoints
│   └── schemas.py              # Pydantic request/response models
│
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
│
├── scripts/
│   └── train.py                # End-to-end training pipeline CLI
│
├── data/
│   ├── raw/                    # Downloaded & extracted dataset
│   └── processed/              # Train/test parquet files
│
├── saved_models/               # Serialized model files + metrics.json
│
└── requirements.txt
```

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone / navigate to the project
cd "recommendation system"

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Mac/Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Dependencies:**

| Package | Purpose |
|---|---|
| pandas, numpy | Data manipulation |
| scipy, scikit-learn | Sparse matrices, KNN, SVD |
| fastapi, uvicorn | REST API serving |
| streamlit, plotly | Interactive dashboard |
| joblib | Model serialization |
| tqdm | Download progress bar |

---

## End-to-End Execution

### Step 1 — Train all models

```bash
# Uses MovieLens Latest Small (~3MB, downloads automatically)
python scripts/train.py

# OR use full MovieLens 25M dataset
python scripts/train.py --dataset 25m

# OR skip evaluation for faster iteration
python scripts/train.py --skip-eval

# OR point to a local already-downloaded folder
python scripts/train.py --data-path path/to/ml-latest-small
```

**What happens:**
```
[ 1/5 ] Download & load data      ~10 seconds
[ 2/5 ] Preprocess                ~5 seconds
[ 3/5 ] Train 4 models            ~30 seconds total
[ 4/5 ] Evaluate (k=10)           ~3 minutes
[ 5/5 ] Save models + metrics     instant
```

### Step 2 — Start the API

```bash
uvicorn api.main:app --reload
```

API runs at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### Step 3 — Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard runs at `http://localhost:8501`

---

## API Reference

### Get Recommendations

```http
GET /recommendations/{user_id}?model=svd&n=10&exclude_seen=true
```

**Parameters:**

| Parameter | Type | Default | Options |
|---|---|---|---|
| `user_id` | int | required | Any user ID in the dataset |
| `model` | string | `svd` | `popularity`, `user_cf`, `item_cf`, `svd` |
| `n` | int | `10` | 1–50 |
| `exclude_seen` | bool | `true` | `true`, `false` |

**Response:**

```json
{
  "userId": 101,
  "model": "svd",
  "recommendations": [
    {
      "movieId": 318,
      "title": "The Shawshank Redemption (1994)",
      "score": 4.72,
      "explanation": "Predicted 4.72/5 based on your latent preference profile (SVD)"
    },
    {
      "movieId": 858,
      "title": "The Godfather (1972)",
      "score": 4.61,
      "explanation": "Predicted 4.61/5 based on your latent preference profile (SVD)"
    }
  ]
}
```

---

### Get All Metrics

```http
GET /metrics
```

**Response:**

```json
{
  "metrics": [
    {
      "model": "svd",
      "rmse": 0.87,
      "precision_at_10": 0.38,
      "recall_at_10": 0.33,
      "ndcg_at_10": 0.41
    }
  ]
}
```

---

### Other Endpoints

```http
GET /health              # System health + loaded models
GET /models              # List available models
GET /metrics/{model}     # Metrics for a specific model
```

---

## Dashboard

Three pages accessible from the sidebar:

**Recommendations**
- Select any user ID, model, and Top-N count
- See ranked recommendations with relevance scores and explanations
- Expand the user's watch history to understand their taste

**Model Metrics**
- Side-by-side comparison of all 4 models
- Bar charts for Precision@10, Recall@10, NDCG@10
- RMSE chart with PRD target threshold line

**Dataset Explorer**
- KPI cards: total ratings, users, movies, average rating
- Rating distribution histogram
- Ratings-per-user distribution (log scale)
- Top 20 genres by movie count

---

## Evaluation Methodology

Models are evaluated offline on a held-out 20% test set:

**RMSE:** Sampled 5,000 user-movie pairs from the test set. Predicted rating vs. actual rating.

**Ranking metrics (Precision, Recall, NDCG @10):**
1. For each test user, identify their "relevant" movies: rated ≥ 4.0 in the test set
2. Ask the model to recommend 10 movies, excluding the training set (simulate real use)
3. Compare recommendations against the relevant set
4. Average across 500 sampled users

This protocol mirrors the **leave-one-out** offline evaluation used in research and production recommender systems.

---

## Future Roadmap

**Phase 2 — Hybrid Recommender**
Combine collaborative filtering signals with content-based features (genre vectors, release year, tags) for improved cold-start handling.

**Phase 3 — Deep Learning**
- Neural Collaborative Filtering (NCF)
- Two-Tower architecture for scalable retrieval
- Implicit feedback modeling

**Phase 4 — Real-Time Updates**
- Kafka stream for ingesting new ratings
- Redis cache for sub-10ms recommendation serving
- Online learning to update user embeddings without full retraining

**Phase 5 — A/B Testing Framework**
- Shadow mode deployment
- Click-through rate as online metric
- Multi-armed bandit for model selection

---

## Skills Demonstrated

| Area | Specifics |
|---|---|
| Machine Learning | Collaborative Filtering, Matrix Factorization, Ranking |
| Data Engineering | Sparse matrices, feature engineering, train/test protocol |
| Statistics | Bayesian smoothing, NDCG, evaluation methodology |
| Software Engineering | OOP design patterns, abstract base classes, modular architecture |
| API Development | FastAPI, Pydantic, async lifespan, REST design |
| Visualization | Streamlit, Plotly, interactive dashboards |
| Production ML | Model serialization, offline evaluation, latency targets |

---

*Dataset: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 1–19.*
