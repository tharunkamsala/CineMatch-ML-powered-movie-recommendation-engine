"""
Shadow mode deployment.
Primary model serves real users. Shadow models run silently in parallel
and log their results for offline comparison — zero impact on latency.
"""
import time
import threading
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("shadow")


class ShadowDeployment:
    """
    Wraps a primary recommender. On each recommend() call:
      1. Returns the primary model's result immediately (no latency added)
      2. Fires shadow model calls in background threads
      3. Logs both outputs for offline A/B analysis
    """

    def __init__(self, primary_model, shadow_models: Dict[str, Any], tracker=None):
        self.primary        = primary_model
        self.shadows        = shadow_models       # {name: model}
        self.tracker        = tracker
        self._shadow_log: List[dict] = []
        self._lock = threading.Lock()

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        log_impression: bool = True,
    ) -> List[Dict[str, Any]]:
        # ── Primary (blocking, returns to user) ──────────────────
        t0 = time.perf_counter()
        result = self.primary.recommend(user_id, n=n, exclude_seen=exclude_seen)
        primary_latency = (time.perf_counter() - t0) * 1000  # ms

        if log_impression and self.tracker and result:
            movie_ids = [r["movieId"] for r in result]
            self.tracker.log_impression(user_id, self.primary.name, movie_ids)

        # ── Shadows (non-blocking) ────────────────────────────────
        def run_shadow(name, model):
            try:
                t_s = time.perf_counter()
                shadow_result = model.recommend(user_id, n=n, exclude_seen=exclude_seen)
                latency_ms    = (time.perf_counter() - t_s) * 1000

                entry = {
                    "user_id":         user_id,
                    "primary_model":   self.primary.name,
                    "shadow_model":    name,
                    "primary_top3":    [r["movieId"] for r in result[:3]],
                    "shadow_top3":     [r["movieId"] for r in shadow_result[:3]],
                    "primary_latency": round(primary_latency, 2),
                    "shadow_latency":  round(latency_ms, 2),
                    "overlap@3":       self._overlap(result[:3], shadow_result[:3]),
                }
                with self._lock:
                    self._shadow_log.append(entry)

                logger.debug(
                    "Shadow [%s vs %s] user=%d overlap@3=%.2f shadow_latency=%.1fms",
                    self.primary.name, name, user_id, entry["overlap@3"], latency_ms
                )
            except Exception as e:
                logger.warning("Shadow model %s failed for user %d: %s", name, user_id, e)

        for name, model in self.shadows.items():
            t = threading.Thread(target=run_shadow, args=(name, model), daemon=True)
            t.start()

        return result

    def shadow_report(self) -> dict:
        """Aggregate shadow log into comparison statistics."""
        with self._lock:
            log = list(self._shadow_log)

        if not log:
            return {"message": "No shadow comparisons recorded yet."}

        by_model: Dict[str, list] = {}
        for entry in log:
            by_model.setdefault(entry["shadow_model"], []).append(entry)

        report = {"total_comparisons": len(log), "models": {}}
        for name, entries in by_model.items():
            overlaps  = [e["overlap@3"] for e in entries]
            s_latency = [e["shadow_latency"] for e in entries]
            p_latency = [e["primary_latency"] for e in entries]
            report["models"][name] = {
                "comparisons":       len(entries),
                "avg_overlap@3":     round(sum(overlaps) / len(overlaps), 3),
                "avg_latency_ms":    round(sum(s_latency) / len(s_latency), 1),
                "primary_latency_ms": round(sum(p_latency) / len(p_latency), 1),
            }
        return report

    @staticmethod
    def _overlap(list_a: List[dict], list_b: List[dict]) -> float:
        ids_a = {r["movieId"] for r in list_a}
        ids_b = {r["movieId"] for r in list_b}
        if not ids_a:
            return 0.0
        return len(ids_a & ids_b) / len(ids_a)
