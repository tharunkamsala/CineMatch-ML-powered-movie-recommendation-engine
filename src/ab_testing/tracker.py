"""
CTR (Click-Through Rate) and experiment tracker.
Stores impressions and clicks in SQLite — lightweight, zero extra services needed.
"""
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Optional


class CTRTracker:
    """
    Tracks which recommendations were shown (impressions) and which
    the user interacted with (clicks/ratings), grouped by model.

    Schema:
        impressions(id, user_id, model, movie_ids, ts)
        clicks(id, user_id, movie_id, model, reward, ts)
    """

    def __init__(self, db_path: Path = Path("saved_models/experiments.db")):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS impressions (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id  INTEGER NOT NULL,
                    model    TEXT    NOT NULL,
                    movie_ids TEXT   NOT NULL,
                    ts       REAL    NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clicks (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id  INTEGER NOT NULL,
                    movie_id INTEGER NOT NULL,
                    model    TEXT    NOT NULL,
                    reward   REAL    NOT NULL DEFAULT 1.0,
                    ts       REAL    NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_imp_model  ON impressions(model)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_clk_model  ON clicks(model)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_imp_user   ON impressions(user_id)")

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    # ── Logging ───────────────────────────────────────────────────

    def log_impression(self, user_id: int, model: str, movie_ids: List[int]) -> int:
        """Record that these movies were shown to the user. Returns impression ID."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO impressions (user_id, model, movie_ids, ts) VALUES (?, ?, ?, ?)",
                (user_id, model, ",".join(map(str, movie_ids)), time.time()),
            )
            return cur.lastrowid

    def log_click(self, user_id: int, movie_id: int, model: str, reward: float = 1.0) -> None:
        """Record that a user interacted with a recommended movie."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO clicks (user_id, movie_id, model, reward, ts) VALUES (?, ?, ?, ?, ?)",
                (user_id, movie_id, model, reward, time.time()),
            )

    # ── Metrics ───────────────────────────────────────────────────

    def ctr(self, model: Optional[str] = None) -> Dict[str, float]:
        """Click-through rate per model (or for a specific model)."""
        with self._conn() as conn:
            if model:
                models = [model]
            else:
                rows = conn.execute("SELECT DISTINCT model FROM impressions").fetchall()
                models = [r[0] for r in rows]

            result = {}
            for m in models:
                imp_count = conn.execute(
                    "SELECT COUNT(*) FROM impressions WHERE model=?", (m,)
                ).fetchone()[0]
                clk_count = conn.execute(
                    "SELECT COUNT(*) FROM clicks WHERE model=?", (m,)
                ).fetchone()[0]
                result[m] = {
                    "impressions": imp_count,
                    "clicks":      clk_count,
                    "ctr":         round(clk_count / imp_count, 4) if imp_count else 0.0,
                }
            return result

    def summary(self) -> dict:
        with self._conn() as conn:
            total_imp = conn.execute("SELECT COUNT(*) FROM impressions").fetchone()[0]
            total_clk = conn.execute("SELECT COUNT(*) FROM clicks").fetchone()[0]
        return {
            "total_impressions": total_imp,
            "total_clicks":      total_clk,
            "overall_ctr":       round(total_clk / total_imp, 4) if total_imp else 0.0,
            "per_model":         self.ctr(),
        }
