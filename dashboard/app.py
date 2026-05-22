import json
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Loaders ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    models = {}
    for name in ["popularity", "user_cf", "item_cf", "svd"]:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_data
def load_metrics():
    path = MODELS_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_processed_data():
    rp = PROCESSED_DATA_DIR / "ratings.parquet"
    mp = PROCESSED_DATA_DIR / "movies.parquet"
    if rp.exists() and mp.exists():
        return pd.read_parquet(rp), pd.read_parquet(mp)
    return None, None


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🎬 Movie Recommender")
st.sidebar.markdown("Powered by Collaborative Filtering & SVD")
st.sidebar.divider()

models = load_models()
metrics = load_metrics()
ratings_df, movies_df = load_processed_data()

if not models:
    st.error("No trained models found. Run `python scripts/train.py` first.")
    st.stop()

page = st.sidebar.radio("Navigate", ["Recommendations", "Model Metrics", "Dataset Explorer"])

# ── Page: Recommendations ─────────────────────────────────────────────────────

if page == "Recommendations":
    st.title("Personalized Movie Recommendations")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    with col2:
        model_name = st.selectbox("Model", list(models.keys()), index=min(3, len(models) - 1))
    with col3:
        n_recs = st.slider("Top N", 5, 20, 10)

    exclude_seen = st.checkbox("Exclude already-watched movies", value=True)

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        model = models[model_name]
        with st.spinner("Generating recommendations..."):
            try:
                recs = model.recommend(user_id, n=n_recs, exclude_seen=exclude_seen)
            except Exception as e:
                st.error(f"Error: {e}")
                recs = []

        if not recs:
            st.warning(f"No recommendations found for user {user_id}. They may not exist in the training set.")
        else:
            st.success(f"Top {len(recs)} recommendations for User {user_id} using **{model_name.upper()}**")

            for i, rec in enumerate(recs, 1):
                with st.container():
                    cols = st.columns([0.5, 5, 2, 4])
                    cols[0].markdown(f"**#{i}**")
                    cols[1].markdown(f"**{rec['title']}**")
                    score = rec["score"]
                    bar_pct = int(min(score / 5.0, 1.0) * 100)
                    cols[2].progress(bar_pct, text=f"{score:.2f}")
                    cols[3].caption(rec.get("explanation", ""))
                st.divider()

    # Show user's watch history
    if ratings_df is not None:
        user_history = ratings_df[ratings_df["userId"] == user_id].merge(
            movies_df[["movieId", "title"]], on="movieId", how="left"
        ).sort_values("rating", ascending=False)

        if not user_history.empty:
            with st.expander(f"Watch history for User {user_id} ({len(user_history)} movies)"):
                st.dataframe(
                    user_history[["title", "rating"]].rename(columns={"title": "Movie", "rating": "Rating"}),
                    hide_index=True,
                    use_container_width=True,
                )

# ── Page: Model Metrics ───────────────────────────────────────────────────────

elif page == "Model Metrics":
    st.title("Model Evaluation Metrics")

    if not metrics:
        st.warning("No metrics found. Run `python scripts/train.py` to evaluate models.")
        st.stop()

    # Summary table
    rows = []
    for model_name, m in metrics.items():
        rows.append({
            "Model": model_name.replace("_", " ").title(),
            "RMSE ↓": m.get("rmse", "—"),
            "Precision@10 ↑": m.get("precision_at_10", "—"),
            "Recall@10 ↑": m.get("recall_at_10", "—"),
            "NDCG@10 ↑": m.get("ndcg_at_10", "—"),
        })
    df_metrics = pd.DataFrame(rows)

    st.subheader("Summary")
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    # Target reference
    targets = {"RMSE ↓": 0.9, "Precision@10 ↑": 0.35, "Recall@10 ↑": 0.30}
    st.caption("Targets: RMSE < 0.9 | Precision@10 > 0.35 | Recall@10 > 0.30")
    st.divider()

    # Bar charts
    col1, col2 = st.columns(2)

    numeric_cols = ["Precision@10 ↑", "Recall@10 ↑", "NDCG@10 ↑"]
    df_plot = df_metrics[["Model"] + numeric_cols].copy()
    df_melt = df_plot.melt(id_vars="Model", var_name="Metric", value_name="Score")
    df_melt["Score"] = pd.to_numeric(df_melt["Score"], errors="coerce")

    with col1:
        fig = px.bar(df_melt, x="Model", y="Score", color="Metric", barmode="group",
                     title="Ranking Metrics Comparison")
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rmse_df = pd.DataFrame([
            {"Model": r["Model"], "RMSE": r["RMSE ↓"]} for r in rows
        ])
        rmse_df["RMSE"] = pd.to_numeric(rmse_df["RMSE"], errors="coerce")
        fig2 = px.bar(rmse_df, x="Model", y="RMSE", title="RMSE Comparison (lower is better)",
                      color="RMSE", color_continuous_scale="RdYlGn_r")
        fig2.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="Target: 0.9")
        st.plotly_chart(fig2, use_container_width=True)

# ── Page: Dataset Explorer ────────────────────────────────────────────────────

elif page == "Dataset Explorer":
    st.title("Dataset Explorer")

    if ratings_df is None or movies_df is None:
        st.warning("Processed data not found. Run `python scripts/train.py` first.")
        st.stop()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Ratings", f"{len(ratings_df):,}")
    k2.metric("Unique Users", f"{ratings_df['userId'].nunique():,}")
    k3.metric("Unique Movies", f"{ratings_df['movieId'].nunique():,}")
    k4.metric("Avg Rating", f"{ratings_df['rating'].mean():.2f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rating Distribution")
        rating_counts = ratings_df["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Count"]
        fig = px.bar(rating_counts, x="Rating", y="Count", title="Distribution of Ratings")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Ratings per User (log scale)")
        user_counts = ratings_df["userId"].value_counts()
        fig2 = px.histogram(user_counts, log_y=True, nbins=50,
                            title="How many movies each user has rated",
                            labels={"value": "Ratings per User", "count": "Number of Users"})
        st.plotly_chart(fig2, use_container_width=True)

    # Genre breakdown
    st.subheader("Top Genres")
    genre_series = movies_df["genres"].dropna().str.split("|").explode()
    genre_counts = genre_series[genre_series != "(no genres listed)"].value_counts().head(20).reset_index()
    genre_counts.columns = ["Genre", "Count"]
    fig3 = px.bar(genre_counts, x="Count", y="Genre", orientation="h",
                  title="Top 20 Genres by Number of Movies")
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)
