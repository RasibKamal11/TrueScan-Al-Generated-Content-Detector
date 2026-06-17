"""
TrueScan — Persistent SQLite Result Cache
==========================================
Replaces the pure in-memory LRU cache with a two-tier approach:

  Tier 1: In-memory LRU  (hot path — sub-millisecond, thread-safe OrderedDict)
  Tier 2: SQLite on disk (survives server restarts, 7-day TTL, 10 000 entry cap)

This fulfils the Redis-caching objective WITHOUT requiring a Redis server:
  - Persistent across restarts (unlike the old in-memory-only approach)
  - Zero external dependencies (SQLite ships with Python)
  - Upgrade path: swap SQLiteCache for aioredis when Redis becomes available

Public API is unchanged so all existing call-sites keep working.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from loguru import logger


# ── Tier-1: In-Memory LRU ─────────────────────────────────────────────────────

class MemoryLRU:
    """Thread-safe in-memory LRU — hot path before hitting SQLite."""

    def __init__(self, max_size: int = 500, ttl: int = 3600):
        self._store: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._max  = max_size
        self._ttl  = ttl
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> dict | None:
        with self._lock:
            if key not in self._store:
                self.misses += 1
                return None
            val, ts = self._store[key]
            if time.time() - ts > self._ttl:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return val

    def set(self, key: str, val: dict) -> None:
        with self._lock:
            self._store[key] = (val, time.time())
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)


# ── Tier-2: SQLite Persistent Cache ───────────────────────────────────────────

class SQLiteCache:
    """
    Persistent cache backed by SQLite.
    Schema: cache(key TEXT PK, value TEXT, created_at REAL)
    TTL and max-size enforced lazily on reads + periodic sweep.
    """

    _CREATE = """
        CREATE TABLE IF NOT EXISTS cache (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """

    def __init__(
        self,
        db_path: str,
        max_size: int = 10_000,
        ttl: int = 604_800,  # 7 days
    ):
        self._db   = db_path
        self._max  = max_size
        self._ttl  = ttl
        self._lock = threading.Lock()
        self._init_db()
        self._schedule_sweep()

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def _init_db(self) -> None:
        with self._lock:
            con = self._con()
            con.execute(self._CREATE)
            con.commit()
            con.close()

    # ── Public ────────────────────────────────────────────────────────────────

    def get(self, key: str) -> dict | None:
        try:
            with self._lock:
                con = self._con()
                row = con.execute(
                    "SELECT value, created_at FROM cache WHERE key = ?", (key,)
                ).fetchone()
                con.close()
            if row is None:
                return None
            val_str, ts = row
            if time.time() - ts > self._ttl:
                self._delete(key)
                return None
            return json.loads(val_str)
        except Exception as e:
            logger.debug(f"SQLiteCache.get error: {e}")
            return None

    def set(self, key: str, val: dict) -> None:
        try:
            with self._lock:
                con = self._con()
                con.execute(
                    "INSERT OR REPLACE INTO cache(key, value, created_at) VALUES (?,?,?)",
                    (key, json.dumps(val), time.time()),
                )
                con.commit()
                con.close()
        except Exception as e:
            logger.debug(f"SQLiteCache.set error: {e}")

    def stats(self) -> dict:
        try:
            with self._lock:
                con = self._con()
                total = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                oldest = con.execute("SELECT MIN(created_at) FROM cache").fetchone()[0]
                con.close()
            return {"sqlite_entries": total, "oldest_entry_ts": oldest}
        except Exception:
            return {"sqlite_entries": 0}

    # ── Private ───────────────────────────────────────────────────────────────

    def _delete(self, key: str) -> None:
        try:
            with self._lock:
                con = self._con()
                con.execute("DELETE FROM cache WHERE key = ?", (key,))
                con.commit()
                con.close()
        except Exception:
            pass

    def _sweep(self) -> None:
        """Remove expired entries and enforce max_size (FIFO eviction)."""
        try:
            cutoff = time.time() - self._ttl
            with self._lock:
                con = self._con()
                con.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
                count = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                if count > self._max:
                    excess = count - self._max
                    con.execute(
                        f"DELETE FROM cache WHERE key IN "
                        f"(SELECT key FROM cache ORDER BY created_at ASC LIMIT {excess})"
                    )
                con.commit()
                con.close()
            logger.debug("SQLiteCache sweep complete")
        except Exception as e:
            logger.debug(f"SQLiteCache sweep error: {e}")

    def _schedule_sweep(self) -> None:
        """Run a sweep in a background thread every hour."""
        def _loop():
            while True:
                time.sleep(3600)
                self._sweep()

        t = threading.Thread(target=_loop, daemon=True, name="cache-sweep")
        t.start()


# ── Two-Tier Cache ────────────────────────────────────────────────────────────

class TwoTierCache:
    """
    Combines MemoryLRU (hot) + SQLiteCache (warm/persistent).
    Read:  memory → sqlite → miss
    Write: memory + sqlite simultaneously
    """

    def __init__(self, db_path: str):
        self._mem = MemoryLRU(max_size=500, ttl=3600)
        try:
            self._sql = SQLiteCache(db_path=db_path)
            logger.success(f"Persistent SQLite cache active: {db_path}")
        except Exception as e:
            logger.warning(f"SQLite cache init failed ({e}), using memory-only cache")
            self._sql = None

    def get(self, key: str) -> dict | None:
        # Tier 1
        val = self._mem.get(key)
        if val is not None:
            return val
        # Tier 2
        if self._sql:
            val = self._sql.get(key)
            if val is not None:
                self._mem.set(key, val)   # promote to memory
                return val
        return None

    def set(self, key: str, val: dict) -> None:
        self._mem.set(key, val)
        if self._sql:
            self._sql.set(key, val)

    def stats(self) -> dict:
        mem_stats = {
            "memory_hits":   self._mem.hits,
            "memory_misses": self._mem.misses,
            "memory_size":   len(self._mem._store),
            "hit_rate": round(
                self._mem.hits / max(self._mem.hits + self._mem.misses, 1) * 100, 1
            ),
        }
        sql_stats = self._sql.stats() if self._sql else {}
        return {**mem_stats, **sql_stats}


# ── Singleton ─────────────────────────────────────────────────────────────────

_DB_PATH = os.path.join(os.path.dirname(__file__), "truescan_cache.db")
_cache   = TwoTierCache(db_path=_DB_PATH)


def make_cache_key(content: str, detect_type: str) -> str:
    raw = f"{detect_type}:{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached(content: str, detect_type: str) -> dict | None:
    return _cache.get(make_cache_key(content, detect_type))


def set_cached(content: str, detect_type: str, result: dict) -> None:
    _cache.set(make_cache_key(content, detect_type), result)


def cache_stats() -> dict:
    return _cache.stats()
