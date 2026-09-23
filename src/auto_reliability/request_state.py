"""Persistent operational state, separate from immutable API evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class RequestState:
    """Track host cooldowns and latest query failures across daily snapshots.

    The caller holds the shared request lock while checking and updating state.
    Transactions preserve cooldown and failure state together. This is not an
    automatic retry worker: recoverable entries are retried by explicit queries.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("CREATE TABLE IF NOT EXISTS cooldowns (host TEXT PRIMARY KEY, until REAL NOT NULL)")
        connection.execute("""CREATE TABLE IF NOT EXISTS requests (
            key TEXT PRIMARY KEY, endpoint TEXT NOT NULL, parameters TEXT NOT NULL,
            state TEXT NOT NULL, status INTEGER, attempts INTEGER NOT NULL,
            updated REAL NOT NULL, retry_at REAL, error_type TEXT)""")
        connection.commit()
        return connection

    def remaining(self, endpoint: str, now: float) -> float:
        """Return persisted minimum server wait without sleeping or networking."""
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT until FROM cooldowns WHERE host=?", (urlsplit(endpoint).netloc,)).fetchone()
        if row is None:
            return 0.0
        until = float(row[0])
        if not math.isfinite(until):
            raise ValueError("Invalid persisted cooldown")
        return max(0.0, until - now)

    def record(self, endpoint: str, params: Mapping[str, object] | None, *, now: float,
               state: str, status: int | None, attempts: int, delay: float = 0,
               error_type: str | None = None) -> None:
        """Record a result and extend (never shorten) a server cooldown."""
        if state not in {"succeeded", "retryable", "permanent_failure"} or attempts < 0:
            raise ValueError("Invalid request state")
        if not math.isfinite(now) or not math.isfinite(delay) or delay < 0:
            raise ValueError("Invalid request timestamps")
        parameters = json.dumps(dict(params or {}), sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256((endpoint + "\n" + parameters).encode()).hexdigest()
        retry_at = now + delay if state == "retryable" else None
        with closing(self._connect()) as connection, connection:
            connection.execute("""INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET state=excluded.state, status=excluded.status,
                attempts=requests.attempts+excluded.attempts, updated=excluded.updated,
                retry_at=excluded.retry_at, error_type=excluded.error_type""",
                (key, endpoint, parameters, state, status, attempts, now, retry_at, error_type))
            if delay:
                connection.execute("""INSERT INTO cooldowns VALUES (?, ?)
                    ON CONFLICT(host) DO UPDATE SET until=MAX(cooldowns.until, excluded.until)""",
                    (urlsplit(endpoint).netloc, now + delay))

    def failures(self) -> list[dict[str, Any]]:
        """Return pending failures for operational inspection, never ML labels."""
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM requests WHERE state != 'succeeded' ORDER BY updated").fetchall()
        return [dict(row) for row in rows]
