"""
Recommendation cache with Redis backend and in-memory fallback.
Drop-in: if Redis is unavailable the system keeps running with local cache.
"""
import json
import time
from typing import Any, Optional


class RecommendationCache:
    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600):
        self.ttl    = ttl
        self._redis = None
        self._local: dict = {}       # {key: (value, expires_at)}
        self._hits  = 0
        self._misses = 0

        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
                self._redis.ping()
                print(f"Cache: connected to Redis at {redis_url}")
            except Exception as e:
                print(f"Cache: Redis unavailable ({e}), using in-memory fallback")
                self._redis = None
        else:
            print("Cache: no Redis URL provided, using in-memory cache")

    # ── Public API ────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        value = self._redis_get(key) if self._redis else self._local_get(key)
        if value is not None:
            self._hits += 1
        else:
            self._misses += 1
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self.ttl
        serialized = json.dumps(value)
        if self._redis:
            self._redis_set(key, serialized, ttl)
        else:
            self._local[key] = (serialized, time.time() + ttl)

    def invalidate(self, key: str) -> None:
        if self._redis:
            self._redis.delete(key)
        else:
            self._local.pop(key, None)

    def invalidate_user(self, user_id: int) -> None:
        """Remove all cached recommendations for a user across all models."""
        prefix = f"rec:{user_id}:"
        if self._redis:
            keys = self._redis.keys(f"{prefix}*")
            if keys:
                self._redis.delete(*keys)
        else:
            stale = [k for k in self._local if k.startswith(prefix)]
            for k in stale:
                del self._local[k]

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(self._hits / total, 4) if total else 0.0,
            "backend":   "redis" if self._redis else "in-memory",
            "size":      len(self._local) if not self._redis else "n/a",
        }

    @staticmethod
    def make_key(user_id: int, model: str, n: int) -> str:
        return f"rec:{user_id}:{model}:{n}"

    # ── Internal helpers ──────────────────────────────────────────

    def _redis_get(self, key: str) -> Optional[Any]:
        raw = self._redis.get(key)
        return json.loads(raw) if raw else None

    def _redis_set(self, key: str, value: str, ttl: int) -> None:
        self._redis.setex(key, ttl, value)

    def _local_get(self, key: str) -> Optional[Any]:
        entry = self._local.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._local[key]
            return None
        return json.loads(value)
