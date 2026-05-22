"""
Neural Collaborative Filtering (NCF)
Combines Generalized Matrix Factorization (GMF) and MLP into a single model.
Paper: He et al., "Neural Collaborative Filtering", WWW 2017.
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.base import BaseRecommender
from src.data.processor import build_index_maps


class RatingDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users  = torch.LongTensor(users)
        self.items  = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class NCFNet(nn.Module):
    """GMF + MLP fusion network."""

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 32, mlp_layers: List[int] = None):
        super().__init__()
        mlp_layers = mlp_layers or [64, 32, 16]

        # GMF embeddings
        self.gmf_user = nn.Embedding(n_users, embed_dim)
        self.gmf_item = nn.Embedding(n_items, embed_dim)

        # MLP embeddings
        self.mlp_user = nn.Embedding(n_users, embed_dim)
        self.mlp_item = nn.Embedding(n_items, embed_dim)

        # MLP tower
        layers = []
        in_size = embed_dim * 2
        for out_size in mlp_layers:
            layers += [nn.Linear(in_size, out_size), nn.BatchNorm1d(out_size), nn.ReLU(), nn.Dropout(0.2)]
            in_size = out_size
        self.mlp = nn.Sequential(*layers)

        # Final fusion layer (GMF output + last MLP layer)
        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # GMF path
        gmf_out = self.gmf_user(user_ids) * self.gmf_item(item_ids)

        # MLP path
        mlp_in = torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Fusion
        x = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output(x).squeeze(-1)


class NCFRecommender(BaseRecommender):
    name = "ncf"

    def __init__(self, embed_dim: int = 32, mlp_layers: List[int] = None,
                 epochs: int = 10, batch_size: int = 512, lr: float = 1e-3):
        self.embed_dim  = embed_dim
        self.mlp_layers = mlp_layers or [64, 32, 16]
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_ratings: pd.DataFrame, movies: pd.DataFrame, **kwargs) -> None:
        self._build_seen_index(train_ratings)
        self._movies = movies.set_index("movieId")

        self._user2idx, self._movie2idx, self._idx2user, self._idx2movie = build_index_maps(train_ratings)

        n_users = len(self._user2idx)
        n_items = len(self._movie2idx)

        users   = train_ratings["userId"].map(self._user2idx).values
        items   = train_ratings["movieId"].map(self._movie2idx).values
        # Normalize ratings to [0, 1] for sigmoid-free regression
        ratings = ((train_ratings["rating"].values - 0.5) / 4.5).astype(np.float32)

        dataset = RatingDataset(users, items, ratings)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.model = NCFNet(n_users, n_items, self.embed_dim, self.mlp_layers).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion  = nn.MSELoss()

        print(f"Training NCF on {self.device} — {n_users} users, {n_items} items, {self.epochs} epochs")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for u, i, r in loader:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                optimizer.zero_grad()
                pred = self.model(u, i)
                loss = criterion(pred, r)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")

        # Precompute all item embeddings for fast retrieval
        self._precompute_item_scores()
        print("NCF training complete.")

    def _precompute_item_scores(self):
        """Cache full prediction matrix for fast recommend() calls."""
        self.model.eval()
        n_users = len(self._user2idx)
        n_items = len(self._movie2idx)
        self._pred_matrix = np.zeros((n_users, n_items), dtype=np.float32)

        batch = 256
        with torch.no_grad():
            for u_start in range(0, n_users, batch):
                u_end = min(u_start + batch, n_users)
                u_ids = torch.arange(u_start, u_end).to(self.device)
                for i_start in range(0, n_items, batch):
                    i_end = min(i_start + batch, n_items)
                    i_ids = torch.arange(i_start, i_end).to(self.device)
                    uu = u_ids.repeat_interleave(i_end - i_start)
                    ii = i_ids.repeat(u_end - u_start)
                    preds = self.model(uu, ii).cpu().numpy()
                    self._pred_matrix[u_start:u_end, i_start:i_end] = preds.reshape(u_end - u_start, i_end - i_start)

        # Scale back to [0.5, 5.0]
        self._pred_matrix = self._pred_matrix * 4.5 + 0.5
        self._pred_matrix = np.clip(self._pred_matrix, 0.5, 5.0)

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> List[Dict[str, Any]]:
        if user_id not in self._user2idx:
            return []

        seen  = self._get_seen(user_id) if exclude_seen else set()
        u_idx = self._user2idx[user_id]
        scores = self._pred_matrix[u_idx].copy()

        for movie_id in seen:
            if movie_id in self._movie2idx:
                scores[self._movie2idx[movie_id]] = -np.inf

        top_idx = np.argsort(scores)[::-1][:n]
        results = []
        for m_idx in top_idx:
            movie_id = self._idx2movie[m_idx]
            score = float(scores[m_idx])
            if score == -np.inf:
                break
            title = self._movies.loc[movie_id, "title"] if movie_id in self._movies.index else "Unknown"
            results.append({
                "movieId": int(movie_id),
                "title":   title,
                "score":   round(score, 4),
                "explanation": f"Neural network predicted {score:.2f}/5 (GMF + MLP fusion)",
            })
        return results

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        u_idx = self._user2idx.get(user_id)
        m_idx = self._movie2idx.get(movie_id)
        if u_idx is None or m_idx is None:
            return 3.5
        return float(self._pred_matrix[u_idx, m_idx])
