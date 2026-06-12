"""
In-memory TTL cache with a Redis-compatible interface.

Usage:
    cache = get_cache()
    cache.set("key", value, ttl=60)
    value = cache.get("key")

To swap to Redis in production:
    Set USE_REDIS=true and REDIS_URL=redis://... in your .env
    The CacheService interface is identical.
"""

from __future__ import annotations
import json
import time
from typing import Any, Optional

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


# ── In-memory implementation ──────────────────────────────────────────────────

class _TTLEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: int) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl if ttl > 0 else float("inf")

    @property
    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class InMemoryCache:
    """Thread-safe enough for a single-process FastAPI app."""

    def __init__(self, max_items: int = 1000, default_ttl: int = 3600) -> None:
        self._store: dict[str, _TTLEntry] = {}
        self._max_items = max_items
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    # ── Redis-compatible interface ─────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None or entry.is_expired:
            if entry:
                del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if len(self._store) >= self._max_items:
            self._evict()
        self._store[key] = _TTLEntry(value, ttl if ttl is not None else self._default_ttl)

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def flush(self) -> None:
        self._store.clear()

    def stats(self) -> dict:
        return {
            "backend": "in-memory",
            "items": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Remove expired entries; if still full, evict 10% oldest."""
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]
        if len(self._store) >= self._max_items:
            to_remove = max(1, self._max_items // 10)
            for k in list(self._store.keys())[:to_remove]:
                del self._store[k]


# ── Redis implementation ───────────────────────────────────────────────────────

class RedisCache:
    """Thin Redis wrapper with the same interface as InMemoryCache."""

    def __init__(self, url: str, default_ttl: int = 3600) -> None:
        try:
            import redis  # type: ignore
            self._client = redis.from_url(url, decode_responses=True)
            self._client.ping()
            log.info("Connected to Redis at %s", url)
        except Exception as exc:
            raise RuntimeError(f"Redis connection failed: {exc}") from exc
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(key)
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._client.setex(
            key,
            ttl if ttl is not None else self._default_ttl,
            json.dumps(value, default=str),
        )

    def delete(self, key: str) -> bool:
        return bool(self._client.delete(key))

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(key))

    def flush(self) -> None:
        self._client.flushdb()

    def stats(self) -> dict:
        info = self._client.info()
        return {
            "backend": "redis",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
            "redis_version": info.get("redis_version"),
            "used_memory_human": info.get("used_memory_human"),
        }


# ── Factory ───────────────────────────────────────────────────────────────────

_cache_instance: Optional[InMemoryCache | RedisCache] = None


def get_cache() -> InMemoryCache | RedisCache:
    global _cache_instance
    if _cache_instance is None:
        if settings.USE_REDIS:
            try:
                _cache_instance = RedisCache(settings.REDIS_URL, settings.CACHE_TTL_SECONDS)
            except RuntimeError:
                log.warning("Redis unavailable — falling back to in-memory cache")
                _cache_instance = InMemoryCache(settings.CACHE_MAX_ITEMS, settings.CACHE_TTL_SECONDS)
        else:
            log.info("Using in-memory TTL cache")
            _cache_instance = InMemoryCache(settings.CACHE_MAX_ITEMS, settings.CACHE_TTL_SECONDS)
    return _cache_instance
