"""Tests for DDL (Data Definition Language) operations."""

import importlib
import os
import sys
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_ddl() -> Any:
    """Import and return DDLOperations with stubs in place."""
    module = importlib.import_module("ssb_parquedit.ddl")
    importlib.reload(module)
    return module.DDLOperations


class TestCreateTable:
    """Table creation from various source types."""

    def test_from_dataframe(self, sut_ddl: Any, fake_conn: MagicMock) -> None:
        ddl_ops = sut_ddl(fake_conn)
        ddl_ops.create_table("users", sys.modules["pandas"].DataFrame())
        assert fake_conn.execute.called

    def test_from_schema_includes_id_column(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        ddl_ops = sut_ddl(fake_conn)
        ddl_ops.create_table("users", {"properties": {"id": {"type": "integer"}}})
        all_calls = [str(c) for c in fake_conn.execute.call_args_list]
        assert any("_id VARCHAR" in c for c in all_calls)

    def test_from_parquet_uses_parameterized_query(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        ddl_ops = sut_ddl(fake_conn)
        ddl_ops.create_table("users", "gs://bucket/data.parquet")
        assert any(len(c[0]) > 1 for c in fake_conn.execute.call_args_list)

    def test_invalid_name_raises_valueerror(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        ddl_ops = sut_ddl(fake_conn)
        with pytest.raises(ValueError, match="Invalid table name"):
            ddl_ops.create_table("123bad", sys.modules["pandas"].DataFrame())

    def test_invalid_source_raises_typeerror(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        ddl_ops = sut_ddl(fake_conn)
        with pytest.raises(TypeError):
            ddl_ops.create_table("users", 12345)


class TestDropTable:
    """Table deletion environment guards."""

    def test_blocked_outside_test_env(
        self, sut_ddl: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        ddl_ops = sut_ddl(fake_conn, db_config)
        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            with pytest.raises(PermissionError, match="only allowed in TEST"):
                ddl_ops.drop_table("users")

    def test_allowed_in_test_env(
        self, sut_ddl: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        ddl_ops = sut_ddl(fake_conn, db_config)
        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            ddl_ops.drop_table("users")
        assert fake_conn.execute.called
