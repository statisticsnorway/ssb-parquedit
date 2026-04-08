"""Tests for DDL (Data Definition Language) operations."""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_ddl() -> Any:
    """Import and return the DDLOperations class."""
    import importlib

    module = importlib.import_module("ssb_parquedit.ddl")
    importlib.reload(module)
    return module.DDLOperations


class TestCreateTableFromDataFrame:
    """Test table creation from DataFrame."""

    def test_create_from_dataframe_basic(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test basic table creation from DataFrame."""
        ddl_ops = sut_ddl(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        ddl_ops.create_table("users", df)

        # Should have called execute
        assert fake_conn.execute.called

    def test_create_from_dataframe_validates_table_name(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated."""
        ddl_ops = sut_ddl(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        with pytest.raises(ValueError, match="Invalid table name"):
            ddl_ops.create_table("123invalid", df)


class TestCreateTableFromParquet:
    """Test table creation from Parquet file."""

    def test_create_from_parquet_basic(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test basic table creation from Parquet file."""
        ddl_ops = sut_ddl(fake_conn)

        ddl_ops.create_table("users", "gs://bucket/users.parquet")

        # Should have called execute
        assert fake_conn.execute.called


class TestCreateTableFromSchema:
    """Test table creation from JSON Schema."""

    def test_create_from_schema_basic(self, sut_ddl: Any, fake_conn: MagicMock) -> None:
        """Test basic table creation from schema."""
        ddl_ops = sut_ddl(fake_conn)

        schema = {
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["id"],
        }

        ddl_ops.create_table("users", schema)

        # Should have called execute
        assert fake_conn.execute.called


class TestCreateTableTypeErrors:
    """Test error handling for invalid source types."""

    def test_create_with_invalid_source_type(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that invalid source type raises TypeError."""
        ddl_ops = sut_ddl(fake_conn)

        with pytest.raises(TypeError, match="source must be a DataFrame"):
            ddl_ops.create_table("users", 12345)

    def test_create_with_list_source_raises_error(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that list source raises TypeError."""
        ddl_ops = sut_ddl(fake_conn)

        with pytest.raises(TypeError, match="source must be a DataFrame"):
            ddl_ops.create_table("users", [1, 2, 3])


class TestDropTable:
    """Test table dropping with environment restrictions."""

    def test_drop_table_succeeds_in_test_environment(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that drop_table succeeds when DAPLA_ENVIRONMENT=test."""
        import os
        from unittest.mock import patch

        ddl_ops = sut_ddl(fake_conn)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            ddl_ops.drop_table("users")

            # Verify execute was called with DROP TABLE
            fake_conn.execute.assert_called()
            call_args = fake_conn.execute.call_args[0][0]
            assert "DROP TABLE users" in call_args

    def test_drop_table_fails_in_prod_environment(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that drop_table raises PermissionError when DAPLA_ENVIRONMENT=prod."""
        import os
        from unittest.mock import patch

        ddl_ops = sut_ddl(fake_conn)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            with pytest.raises(PermissionError, match="only allowed in TEST"):
                ddl_ops.drop_table("users")
