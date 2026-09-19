"""SQLite inventory of downloaded/converted files.

The manifest is what makes the backfill idempotent: re-running plans only what
is missing or previously failed. Raw zips are deleted after conversion, so the
manifest also retains each file's SHA256 as the integrity record.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    key          TEXT PRIMARY KEY,   -- object key on data.binance.vision
    dataset      TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    period       TEXT NOT NULL,      -- YYYY-MM (monthly) or YYYY-MM-DD (daily)
    granularity  TEXT NOT NULL,      -- 'monthly' | 'daily'
    status       TEXT NOT NULL,      -- 'converted' | 'failed'
    sha256       TEXT,
    rows         INTEGER,
    parquet_path TEXT,
    error        TEXT,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_dataset_symbol ON files(dataset, symbol);
"""


class Manifest:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def converted_keys(self) -> Set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key FROM files WHERE status = 'converted'"
            ).fetchall()
        return {r[0] for r in rows}

    def record(self, key: str, dataset: str, symbol: str, period: str,
               granularity: str, status: str, sha256: Optional[str] = None,
               rows: Optional[int] = None, parquet_path: Optional[str] = None,
               error: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                """INSERT INTO files
                   (key, dataset, symbol, period, granularity, status,
                    sha256, rows, parquet_path, error, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     status = excluded.status,
                     sha256 = excluded.sha256,
                     rows = excluded.rows,
                     parquet_path = excluded.parquet_path,
                     error = excluded.error,
                     updated_at = excluded.updated_at""",
                (key, dataset, symbol, period, granularity, status,
                 sha256, rows, parquet_path, error, now),
            )
            self._conn.commit()

    def summary(self) -> str:
        with self._lock:
            rows = self._conn.execute(
                """SELECT dataset, status, COUNT(*), COALESCE(SUM(rows), 0)
                   FROM files GROUP BY dataset, status ORDER BY dataset, status"""
            ).fetchall()
        lines = [f"{ds:22s} {st:10s} files={n:6d} rows={total:,}"
                 for ds, st, n, total in rows]
        return "\n".join(lines) if lines else "(manifest empty)"

    def close(self) -> None:
        with self._lock:
            self._conn.close()
