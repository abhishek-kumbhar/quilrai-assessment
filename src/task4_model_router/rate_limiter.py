"""Token-aware sliding-window rate limiter backed by an on-disk SQLite file.

Each request stores a row (tenant, tokens, timestamp). A request is admitted if the
tokens of the tenant's rows from the last `window_seconds`, plus its own estimate,
stay within the limit. Rows older than the window are deleted on every reservation.
"""

import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager

DEFAULT_TOKENS_PER_MINUTE = 50_000


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: float):
        super().__init__(f"rate limit exceeded, retry after {retry_after_seconds:.1f}s")
        self.retry_after_seconds = retry_after_seconds


class TokenRateLimiter:
    def __init__(
        self,
        db_path: str,
        limit_tokens: int = DEFAULT_TOKENS_PER_MINUTE,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ):
        self.db_path = db_path
        self.limit_tokens = limit_tokens
        self.window_seconds = window_seconds
        self.clock = clock
        with self._transaction() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant TEXT NOT NULL,
                    tokens INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_tenant_time ON token_usage (tenant, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_time ON token_usage (created_at)")
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

    def _connect(self) -> sqlite3.Connection:
        # A connection per operation keeps this safe to call from worker threads.
        # isolation_level=None lets us issue BEGIN IMMEDIATE ourselves.
        return sqlite3.connect(self.db_path, timeout=10, isolation_level=None)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            # IMMEDIATE takes the write lock up front, so two requests (threads or processes)
            # can't both read the same usage total and both decide they fit.
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def reserve(self, tenant: str, tokens: int) -> int:
        """Admit a request for `tokens` or raise RateLimitExceeded. Returns a reservation id."""
        with self._transaction() as conn:
            # Read the clock only once the write lock is held; time spent waiting for it
            # would otherwise make the window boundary and the row's timestamp stale.
            now = self.clock()
            window_start = now - self.window_seconds
            conn.execute("DELETE FROM token_usage WHERE created_at <= ?", (window_start,))
            rows = conn.execute(
                "SELECT tokens, created_at FROM token_usage WHERE tenant = ? ORDER BY created_at",
                (tenant,),
            ).fetchall()
            used = sum(row_tokens for row_tokens, _ in rows)
            if used + tokens > self.limit_tokens:
                raise RateLimitExceeded(self._retry_after(rows, used, tokens, now))

            cursor = conn.execute(
                "INSERT INTO token_usage (tenant, tokens, created_at) VALUES (?, ?, ?)", (tenant, tokens, now)
            )
            return cursor.lastrowid

    def _retry_after(self, rows: list[tuple[int, float]], used: int, tokens: int, now: float) -> float:
        # Walk the oldest entries until enough of them would have expired.
        for row_tokens, created_at in rows:
            used -= row_tokens
            if used + tokens <= self.limit_tokens:
                return max(created_at + self.window_seconds - now, 0.0)
        return self.window_seconds

    def settle(self, reservation_id: int, actual_tokens: int) -> None:
        """Replace the estimate with the real token count once the provider reports it."""
        with self._transaction() as conn:
            conn.execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (actual_tokens, reservation_id))

    def release(self, reservation_id: int) -> None:
        with self._transaction() as conn:
            conn.execute("DELETE FROM token_usage WHERE id = ?", (reservation_id,))

    def tokens_used(self, tenant: str) -> int:
        window_start = self.clock() - self.window_seconds
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND created_at > ?",
                (tenant, window_start),
            ).fetchone()
        return row[0]
