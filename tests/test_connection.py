"""Tests for DuckDB connection management."""

from typing import Any
from unittest.mock import MagicMock

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_connection() -> Any:
    """Import and return the DuckDBConnection class."""
    import importlib

    module = importlib.import_module("ssb_parquedit.connection")
    importlib.reload(module)
    return module.DuckDBConnection


class TestDuckDBConnectionExecute:
    """Test execute method."""

    def test_execute_without_parameters(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test execute method without parameters."""
        conn = sut_connection(db_config, fake_conn)

        conn.execute("SELECT * FROM users")

        # When parameters is None, only the SQL is passed
        fake_conn.execute.assert_called_with("SELECT * FROM users")

    def test_execute_with_parameters(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test execute method with parameters."""
        conn = sut_connection(db_config, fake_conn)

        params = [42, "test"]
        conn.execute("SELECT * FROM users WHERE id = ? AND name = ?", params)

        fake_conn.execute.assert_called_with(
            "SELECT * FROM users WHERE id = ? AND name = ?", params
        )

    def test_execute_returns_result(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test that execute returns the result from underlying connection."""
        fake_result = MagicMock()
        fake_conn.execute.return_value = fake_result

        conn = sut_connection(db_config, fake_conn)
        result = conn.execute("SELECT 1")

        assert result == fake_result


class TestDuckDBConnectionSql:
    """Test sql method."""

    def test_sql_executes_query(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test sql method executes query."""
        conn = sut_connection(db_config, fake_conn)

        conn.sql("SELECT * FROM users")

        assert any(
            "SELECT * FROM users" in str(call_args[0])
            for (call_args, _) in fake_conn.sql.call_args_list
        )


class TestDuckDBConnectionRegister:
    """Test register method for virtual tables."""

    def test_register_python_Any(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test registering a Python object as virtual table."""
        conn = sut_connection(db_config, fake_conn)

        obj = MagicMock()
        conn.register("my_table", obj)

        fake_conn.register.assert_called_with("my_table", obj)


class TestDuckDBConnectionClose:
    """Test connection closing behavior."""

    def test_close_when_owns_connection(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that close calls underlying close when connection is owned."""
        # Don't pass fake_conn so ownership is True
        conn = sut_connection(db_config)

        conn.close()

        # Check that the owned connection's close was called
        conn._conn.close.assert_called_once()

    def test_close_when_does_not_own_connection(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test that close does not call underlying close when connection is not owned."""
        # Create with existing connection
        conn = sut_connection(db_config, fake_conn)

        # Reset the mock since init already called sql methods
        fake_conn.reset_mock()

        conn.close()

        # close() should not have been called since we don't own the connection
        fake_conn.close.assert_not_called()


class TestDropOperationEnforcement:
    """Test DROP operation environment-based enforcement."""

    def test_execute_blocks_drop_in_prod_environment(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test that execute() blocks DROP TABLE in prod environment."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config, fake_conn)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            with pytest.raises(PermissionError, match="only allowed in TEST"):
                conn.execute("DROP TABLE users")

    def test_execute_allows_drop_in_test_environment(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test that execute() allows DROP TABLE in test environment."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config, fake_conn)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            # Should not raise
            conn.execute("DROP TABLE users")
            # Verify execute was called
            fake_conn.execute.assert_called()
