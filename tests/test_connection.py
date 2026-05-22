"""Tests for DuckDBConnection - happy path and documented error behavior."""

import pandas as pd
import pytest

from tests.conftest import LocalDuckDBConnection

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


# ── DROP outside TEST environment raises PermissionError ─────────────────────


class TestDropOperationEnforcement:
    """DROP statements must be blocked in every non-test environment."""

    @pytest.mark.parametrize(
        "statement",
        [
            "DROP TABLE my_table",
            "DROP VIEW my_view",
            "DROP DATABASE my_db",
            "DROP SCHEMA my_schema",
            "drop table my_table",  # lower-case keywords
            "DROP TABLE IF EXISTS my_table",  # IF EXISTS variant
        ],
    )
    def test_drop_raises_permission_error_in_prod(
        self,
        conn: LocalDuckDBConnection,
        statement: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DAPLA_ENVIRONMENT", "prod")
        with pytest.raises(PermissionError):
            conn.execute(statement)

    def test_permission_error_names_current_environment(
        self,
        conn: LocalDuckDBConnection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The error message should tell the user which environment they are in."""
        monkeypatch.setenv("DAPLA_ENVIRONMENT", "prod")
        with pytest.raises(PermissionError, match="prod"):
            conn.execute("DROP TABLE my_table")

    def test_permission_error_guides_user_to_fix(
        self,
        conn: LocalDuckDBConnection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The error message should tell the user how to enable DROP operations."""
        monkeypatch.setenv("DAPLA_ENVIRONMENT", "prod")
        with pytest.raises(PermissionError, match="DAPLA_ENVIRONMENT=test"):
            conn.execute("DROP TABLE my_table")

    def test_drop_does_not_raise_permission_error_in_test_environment(
        self,
        conn: LocalDuckDBConnection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In the test environment, DROP must not be blocked by PermissionError.

        A DuckDB catalog error (table does not exist) is acceptable here
        because we are only verifying the permission check, not a full roundtrip.
        """
        monkeypatch.setenv("DAPLA_ENVIRONMENT", "test")
        try:
            conn.execute("DROP TABLE non_existent_table")
        except PermissionError:
            pytest.fail(
                "DROP TABLE must not raise PermissionError when "
                "DAPLA_ENVIRONMENT is 'test'"
            )
        except Exception:
            pass  # DuckDB's own error about missing table is acceptable


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
