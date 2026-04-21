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

    def test_init_with_no_existing_connection(
        self, sut_connection: Any, db_config: dict[str, str]
    ) -> None:
        """Test initialization when no connection is provided (owns connection)."""
        conn = sut_connection(db_config)

        # Should own the connection
        assert conn.owns_connection is True
        # _owns_conn should be True
        assert conn._owns_conn is True

    def test_init_with_existing_connection(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test initialization with existing connection (doesn't own it)."""
        conn = sut_connection(db_config, fake_conn)

        # Should not own the connection
        assert conn.owns_connection is False
        # _owns_conn should be False
        assert conn._owns_conn is False

    def test_init_registers_gcs_filesystem(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test that GCS filesystem is registered during init."""
        sut_connection(db_config, fake_conn)

        # Should call register_filesystem
        assert fake_conn.register_filesystem.called

    def test_init_loads_extensions(
        self, sut_connection: Any, db_config: dict[str, str], fake_conn: MagicMock
    ) -> None:
        """Test that extensions are loaded during init."""
        sut_connection(db_config, fake_conn)

        # Should install and load ducklake and postgres extensions
        expected_calls = [
            call("INSTALL ducklake"),
            call("LOAD ducklake"),
            call("INSTALL postgres"),
            call("LOAD postgres"),
        ]
        fake_conn.sql.assert_has_calls(expected_calls, any_order=False)
