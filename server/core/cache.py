"""
In-memory TTL cache — single-process, no Redis required.

Two singletons are exported:
    market_cache  — 60-second TTL for market params snapshots
    sim_cache     — 1-hour TTL for completed simulation results
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple


class TTLCache:
    """Thread-safe* key-value store with per-entry TTL.

    * Python's GIL makes dict operations atomic enough for single-process use.
    """

    def __init__(self, default_ttl_seconds: float = 300) -> None:
        self._default_ttl = default_ttl_seconds
        self._store: Dict[str, Tuple[Any, float]] = {}   # key → (value, expires_at)

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value, or None if absent / expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store *value* under *key* with an optional per-entry TTL override."""
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        self._store[key] = (value, expires_at)

    def invalidate(self, key: str) -> None:
        """Remove a single key (no-op if absent)."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Evict all entries."""
        self._store.clear()

    def __len__(self) -> int:
        now = time.monotonic()
        return sum(1 for _, exp in self._store.values() if exp > now)


# ── Module-level singletons ───────────────────────────────────────────────────

market_cache = TTLCache(default_ttl_seconds=60)     # price snapshots: 1-min TTL
sim_cache    = TTLCache(default_ttl_seconds=3600)   # completed results: 1-hr TTL
