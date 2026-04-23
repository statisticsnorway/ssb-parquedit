"""Tests for DuckDB connection management."""

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import call

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_connection() -> Any:
    """Import and return the DuckDBConnection class."""
    import importlib

    module = importlib.import_module("ssb_parquedit.connection")
    importlib.reload(module)
    return module.DuckDBConnection


class TestDuckDBConnectionInit:
    """Test DuckDB connection initialization."""

    def test_init_creates_connection(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that initialization creates an internal DuckDB connection."""
        conn = sut_connection(db_config)

        assert conn._conn is not None

    def test_init_registers_gcs_filesystem(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that GCS filesystem is registered during init."""
        conn = sut_connection(db_config)

        assert conn._conn.register_filesystem.called

    def test_init_loads_extensions(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that extensions are loaded during init."""
        conn = sut_connection(db_config)

        expected_calls = [
            call("INSTALL ducklake"),
            call("LOAD ducklake"),
            call("INSTALL postgres"),
            call("LOAD postgres"),
        ]
        conn._conn.sql.assert_has_calls(expected_calls, any_order=False)

    def test_init_attaches_catalog(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that catalog is attached during init."""
        conn = sut_connection(db_config)

        attach_called = any(
            "ATTACH 'ducklake:postgres" in str(call_args[0])
            for (call_args, _) in conn._conn.sql.call_args_list
        )
        assert attach_called, "ATTACH catalog call not found"

    def test_init_uses_catalog(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that catalog is used (USE command)."""
        conn = sut_connection(db_config)

        use_called = any(
            f"USE {db_config['catalog_name']}" in str(call_args[0])
            for (call_args, _) in conn._conn.sql.call_args_list
        )
        assert use_called, "USE catalog call not found"

    def test_catalog_config_includes_data_path(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that catalog config includes DATA_PATH."""
        conn = sut_connection(db_config)

        data_path_found = any(
            db_config["data_path"] in str(call_args[0])
            for (call_args, _) in conn._conn.sql.call_args_list
        )
        assert data_path_found, "DATA_PATH not found in catalog config"

    def test_catalog_config_includes_metadata_schema(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that catalog config includes METADATA_SCHEMA."""
        conn = sut_connection(db_config)

        schema_found = any(
            db_config["metadata_schema"] in str(call_args[0])
            for (call_args, _) in conn._conn.sql.call_args_list
        )
        assert schema_found, "METADATA_SCHEMA not found in catalog config"


class TestDuckDBConnectionExecute:
    """Test execute method."""

    def test_execute_without_parameters(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test execute method without parameters."""
        conn = sut_connection(db_config)

        conn.execute("SELECT * FROM users")

        # When parameters is None, only the SQL is passed
        conn._conn.execute.assert_called_with("SELECT * FROM users")

    def test_execute_with_parameters(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test execute method with parameters."""
        conn = sut_connection(db_config)

        params = [42, "test"]
        conn.execute("SELECT * FROM users WHERE id = ? AND name = ?", params)

        conn._conn.execute.assert_called_with(
            "SELECT * FROM users WHERE id = ? AND name = ?", params
        )

    def test_execute_returns_result(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that execute returns the result from underlying connection."""
        conn = sut_connection(db_config)
        fake_result = MagicMock()
        conn._conn.execute.return_value = fake_result

        result = conn.execute("SELECT 1")

        assert result == fake_result


class TestDuckDBConnectionSql:
    """Test sql method."""

    def test_sql_executes_query(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test sql method executes query."""
        conn = sut_connection(db_config)
        conn._conn.sql.reset_mock()  # clear calls made during init

        conn.sql("SELECT * FROM users")

        assert any(
            "SELECT * FROM users" in str(call_args[0])
            for (call_args, _) in conn._conn.sql.call_args_list
        )

    def test_sql_returns_result(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that sql returns the result from underlying connection."""
        conn = sut_connection(db_config)
        fake_result = MagicMock()
        conn._conn.sql.return_value = fake_result

        result = conn.sql("SELECT 1")

        assert result == fake_result


class TestDuckDBConnectionRegister:
    """Test register method for virtual tables."""

    def test_register_python_Any(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test registering a Python object as virtual table."""
        conn = sut_connection(db_config)

        obj = MagicMock()
        conn.register("my_table", obj)

        conn._conn.register.assert_called_with("my_table", obj)


class TestDuckDBConnectionClose:
    """Test connection closing behavior."""

    def test_close_when_owns_connection(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that close calls underlying close and nullifies _conn."""
        conn = sut_connection(db_config)
        inner = conn._conn  # save reference before close

        conn.close()

        inner.close.assert_called_once()
        assert conn._conn is None

    def test_close_is_idempotent(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that calling close twice does not raise."""
        conn = sut_connection(db_config)
        conn.close()
        conn.close()  # should not raise

    def test_raw_returns_underlying_connection(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that raw property returns the underlying DuckDB connection."""
        conn = sut_connection(db_config)
        assert conn.raw is conn._conn

    def test_raw_raises_after_close(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that raw raises RuntimeError after connection is closed."""
        conn = sut_connection(db_config)
        conn.close()
        with pytest.raises(RuntimeError, match="Connection is closed"):
            _ = conn.raw


class TestDropOperationEnforcement:
    """Test DROP operation environment-based enforcement."""

    def test_execute_blocks_drop_in_prod_environment(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that execute() blocks DROP TABLE in prod environment."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            with pytest.raises(PermissionError, match="only allowed in TEST"):
                conn.execute("DROP TABLE users")

    def test_execute_allows_drop_in_test_environment(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that execute() allows DROP TABLE in test environment."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            # Should not raise
            conn.execute("DROP TABLE users")
            # Verify execute was called
            conn._conn.execute.assert_called()

    def test_sql_blocks_drop_in_prod_environment(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that sql() blocks DROP TABLE in prod environment."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            with pytest.raises(PermissionError, match="only allowed in TEST"):
                conn.sql("DROP TABLE users")

    def test_sql_allows_drop_in_test_environment(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that sql() allows DROP TABLE in test environment."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            # Should not raise
            conn.sql("DROP TABLE users")
            # Verify sql was called
            conn._conn.sql.assert_called()

    def test_drop_view_also_blocked_in_prod(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that DROP VIEW is also blocked in prod environment."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            with pytest.raises(PermissionError, match="only allowed in TEST"):
                conn.execute("DROP VIEW my_view")

    def test_drop_database_also_blocked_in_prod(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that DROP DATABASE is also blocked in prod environment."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            with pytest.raises(PermissionError, match="only allowed in TEST"):
                conn.execute("DROP DATABASE mydb")

    def test_non_drop_operations_allowed_in_all_environments(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that non-DROP operations are allowed in all environments."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config)
        conn._conn.execute.reset_mock()  # Reset after init to count only the test calls

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            # SELECT, INSERT, UPDATE, DELETE should all work
            conn.execute("SELECT * FROM users")
            conn.execute("INSERT INTO users VALUES (1, 'Alice')")
            conn.execute("UPDATE users SET name = 'Bob'")
            conn.execute("DELETE FROM users WHERE id = 1")

            assert conn._conn.execute.call_count == 4

    def test_drop_logs_in_test_environment(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that DROP operations are logged in test environment."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            with patch("ssb_parquedit.connection.logger") as mock_logger:
                conn.execute("DROP TABLE users")
                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                # Verify log contains important info
                log_msg = mock_logger.warning.call_args[0][0]
                assert "DROP" in log_msg
                assert "users" in log_msg
                assert "TEST" in log_msg

    def test_drop_case_insensitive(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test that DROP keyword matching is case-insensitive."""
        import os
        from unittest.mock import patch

        conn = sut_connection(db_config)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            # Test various case variations
            with pytest.raises(PermissionError):
                conn.execute("drop table users")
            with pytest.raises(PermissionError):
                conn.execute("Drop Table users")
            with pytest.raises(PermissionError):
                conn.execute("DROP table USERS")
