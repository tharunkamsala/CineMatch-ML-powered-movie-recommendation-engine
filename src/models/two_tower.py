"""
Two-Tower (Dual Encoder) Architecture for scalable retrieval.
User tower and Item tower produce embeddings independently.
Similarity = dot product of user embedding and item embedding.
This decoupling allows pre-computing all item embeddings offline.
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.base import BaseRecommender
from src.data.processor import build_index_maps


class ImplicitDataset(Dataset):
    """Treats any rating as a positive interaction (implicit feedback)."""
    def __init__(self, users, items, n_items: int, neg_ratio: int = 4):
        self.users    = users
        self.items    = items
        self.n_items  = n_items
        self.neg_ratio = neg_ratio
        self._build_seen(users, items)

    def _build_seen(self, users, items):
        self._seen = {}
        for u, i in zip(users, items):
            self._seen.setdefault(u, set()).add(i)

    def __len__(self):
        return len(self.users) * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.neg_ratio)
        is_neg  = (idx % (1 + self.neg_ratio)) != 0

        u = int(self.users[pos_idx])
        if not is_neg:
            return torch.tensor(u), torch.tensor(int(self.items[pos_idx])), torch.tensor(1.0)

        # Negative sample: random item not in user's history
        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in self._seen.get(u, set()):
                return torch.tensor(u), torch.tensor(neg_item), torch.tensor(0.0)


class Tower(nn.Module):
    def __init__(self, n_entities: int, embed_dim: int, output_dim: int):
        super().__init__()
        self.embed = nn.Embedding(n_entities, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(),
            nn.Linear(embed_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, ids):
        return self.net(self.embed(ids))


class TwoTowerNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.user_tower = Tower(n_users, embed_dim, output_dim)
        self.item_tower = Tower(n_items, embed_dim, output_dim)

    def forward(self, user_ids, item_ids):
        u_emb = self.user_tower(user_ids)
        i_emb = self.item_tower(item_ids)
        return (u_emb * i_emb).sum(dim=-1)  # dot product similarity

    def get_user_embedding(self, user_ids):
        return self.user_tower(user_ids)

    def get_item_embedding(self, item_ids):
        return self.item_tower(item_ids)


class TwoTowerRecommender(BaseRecommender):
    """
    Two-Tower retrieval model with implicit feedback.
    At serve time: ANN search over pre-computed item embeddings.
    """
    name = "two_tower"

    def __init__(self, embed_dim: int = 64, output_dim: int = 32,
                 epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3):
        self.embed_dim  = embed_dim
        self.output_dim = output_dim
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

        users = train_ratings["userId"].map(self._user2idx).values
        items = train_ratings["movieId"].map(self._movie2idx).values

        dataset = ImplicitDataset(users, items, n_items, neg_ratio=4)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                             num_workers=0, collate_fn=lambda x: x)

        self.model = TwoTowerNet(n_users, n_items, self.embed_dim, self.output_dim).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion  = nn.BCEWithLogitsLoss()

        print(f"Training Two-Tower on {self.device} — {n_users} users, {n_items} items")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches  = 0
            batch_u, batch_i, batch_r = [], [], []
            for sample in loader:
                for u, i, r in sample:
                    batch_u.append(u); batch_i.append(i); batch_r.append(r)
                if len(batch_u) >= self.batch_size:
                    u_t = torch.stack(batch_u).to(self.device)
                    i_t = torch.stack(batch_i).to(self.device)
                    r_t = torch.stack(batch_r).to(self.device)
                    optimizer.zero_grad()
                    pred = self.model(u_t, i_t)
                    loss = criterion(pred, r_t)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches  += 1
                    batch_u, batch_i, batch_r = [], [], []
            if (epoch + 1) % 2 == 0 or epoch == 0:
                avg = total_loss / max(n_batches, 1)
                print(f"  Epoch {epoch+1}/{self.epochs}  loss={avg:.4f}")

        # Pre-compute all item embeddings → ANN index for fast retrieval
        print("Building ANN index over item embeddings...")
        self._build_item_index()
        print("Two-Tower training complete.")

    def _build_item_index(self):
        self.model.eval()
        n_items = len(self._movie2idx)
        all_ids = torch.arange(n_items).to(self.device)
        with torch.no_grad():
            self._item_embeddings = self.model.get_item_embedding(all_ids).cpu().numpy()

        self._ann = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        self._ann.fit(self._item_embeddings)

    def _get_user_embedding(self, user_id: int) -> np.ndarray:
        u_idx = self._user2idx.get(user_id)
        if u_idx is None:
            return None
        self.model.eval()
        with torch.no_grad():
            uid_t = torch.tensor([u_idx]).to(self.device)
            return self.model.get_user_embedding(uid_t).cpu().numpy()

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> List[Dict[str, Any]]:
        u_emb = self._get_user_embedding(user_id)
        if u_emb is None:
            return []

        seen = self._get_seen(user_id) if exclude_seen else set()
        # Retrieve more than n to account for filtering
        k = min(n + len(seen) + 20, len(self._movie2idx))
        distances, indices = self._ann.kneighbors(u_emb, n_neighbors=k)
        distances, indices = distances[0], indices[0]

        results = []
        for dist, m_idx in zip(distances, indices):
            movie_id = self._idx2movie[m_idx]
            if movie_id in seen:
                continue
            score = float(1 - dist)  # cosine similarity
            title = self._movies.loc[movie_id, "title"] if movie_id in self._movies.index else "Unknown"
            results.append({
                "movieId": int(movie_id),
                "title":   title,
                "score":   round(score, 4),
                "explanation": f"Two-Tower retrieval — cosine similarity {score:.3f} in embedding space",
            })
            if len(results) == n:
                break
        return results

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        u_emb = self._get_user_embedding(user_id)
        if u_emb is None:
            return 3.5
        m_idx = self._movie2idx.get(movie_id)
        if m_idx is None:
            return 3.5
        i_emb = self._item_embeddings[m_idx]
        sim = float(np.dot(u_emb[0], i_emb) / (np.linalg.norm(u_emb) * np.linalg.norm(i_emb) + 1e-8))
        return float(np.clip(sim * 4.5 + 0.5, 0.5, 5.0))
