"""Tests for DuckDBConnection - happy path and documented error behavior."""

import pandas as pd
import pytest

from ssb_parquedit.local import LocalDuckDBConnection

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def closed_conn(conn: LocalDuckDBConnection) -> LocalDuckDBConnection:
    """A connection that has already been closed."""
    conn.close()
    return conn


# ── Closed connection raises RuntimeError ────────────────────────────────────


class TestClosedConnectionRaisesRuntimeError:
    """After close(), every public method must raise RuntimeError."""

    def test_execute_raises(self, closed_conn: LocalDuckDBConnection) -> None:
        with pytest.raises(RuntimeError, match=r"Connection is closed\."):
            closed_conn.execute("SELECT 1")

    def test_sql_raises(self, closed_conn: LocalDuckDBConnection) -> None:
        with pytest.raises(RuntimeError, match=r"Connection is closed\."):
            closed_conn.sql("SELECT 1")

    def test_register_raises(self, closed_conn: LocalDuckDBConnection) -> None:
        with pytest.raises(RuntimeError, match=r"Connection is closed\."):
            closed_conn.register("staging", [])

    def test_raw_property_raises(self, closed_conn: LocalDuckDBConnection) -> None:
        with pytest.raises(RuntimeError, match=r"Connection is closed\."):
            _ = closed_conn.raw


# ── Happy path ───────────────────────────────────────────────────────────────


class TestConnectionHappyPath:
    """Core connection operations that must work under normal conditions."""

    def test_execute_returns_query_result(self, conn: LocalDuckDBConnection) -> None:
        result = conn.execute("SELECT 42 AS answer")
        assert result.fetchone()[0] == 42

    def test_sql_returns_query_result(self, conn: LocalDuckDBConnection) -> None:
        result = conn.sql("SELECT 'hello' AS greeting")
        assert result.fetchone()[0] == "hello"

    def test_register_makes_dataframe_queryable(
        self, conn: LocalDuckDBConnection
    ) -> None:
        df = pd.DataFrame({"x": [10, 20, 30]})
        conn.register("tmp_df", df)
        count = conn.execute("SELECT COUNT(*) FROM tmp_df").fetchone()[0]
        assert count == 3

    def test_raw_returns_underlying_connection(
        self, conn: LocalDuckDBConnection
    ) -> None:
        assert conn.raw is not None

    def test_close_twice_does_not_raise(self, conn: LocalDuckDBConnection) -> None:
        conn.close()
        conn.close()  # Second close must be a no-op, not an error
