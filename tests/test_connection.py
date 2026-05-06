"""Tests for DuckDB connection management."""

import importlib
import os
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_connection() -> Any:
    """Import and return DuckDBConnection with stubs in place."""
    module = importlib.import_module("ssb_parquedit.connection")
    importlib.reload(module)
    return module.DuckDBConnection


class TestInit:
    """Connection initialization."""

    def test_creates_underlying_connection(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        conn = sut_connection(db_config)
        assert conn._conn is not None

    def test_installs_extensions(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        conn = sut_connection(db_config)
        calls = [str(c) for c in conn._conn.sql.call_args_list]
        assert any("INSTALL ducklake" in c for c in calls)
        assert any("INSTALL postgres" in c for c in calls)


class TestExecute:
    """execute() method."""

    def test_passes_sql_to_underlying_connection(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        conn = sut_connection(db_config)
        conn.execute("SELECT 1")
        conn._conn.execute.assert_called_with("SELECT 1")

    def test_raises_when_closed(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        conn = sut_connection(db_config)
        conn.close()
        with pytest.raises(RuntimeError, match="Connection is closed"):
            conn.execute("SELECT 1")


class TestSql:
    """sql() method."""

    def test_returns_result_from_underlying_connection(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        conn = sut_connection(db_config)
        fake_result = MagicMock()
        conn._conn.sql.return_value = fake_result
        assert conn.sql("SELECT 1") == fake_result


class TestClose:
    """Connection lifecycle."""

    def test_close_nullifies_conn(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        conn = sut_connection(db_config)
        conn.close()
        assert conn._conn is None


class TestDropEnforcement:
    """DROP operation environment guards."""

    def test_drop_blocked_outside_test_env(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        conn = sut_connection(db_config)
        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            with pytest.raises(PermissionError, match="only allowed in TEST"):
                conn.execute("DROP TABLE users")

    def test_drop_allowed_in_test_env(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        conn = sut_connection(db_config)
        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            conn.execute("DROP TABLE users")
            assert conn._conn.execute.called
