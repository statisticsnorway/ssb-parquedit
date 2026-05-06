"""Tests for DML (Data Manipulation Language) operations."""

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_dml() -> Any:
    """Import and return DMLOperations with stubs in place."""
    module = importlib.import_module("ssb_parquedit.dml")
    importlib.reload(module)
    return module.DMLOperations


class TestInsertData:
    """Data insertion from various source types."""

    def test_from_dataframe_registers_and_inserts(
        self, sut_dml: Any, fake_conn: MagicMock
    ) -> None:
        dml_ops = sut_dml(fake_conn)
        dml_ops.insert_data("users", sys.modules["pandas"].DataFrame())
        assert fake_conn.register.called
        assert any(
            "INSERT INTO users" in str(c) for c in fake_conn.execute.call_args_list
        )

    def test_from_parquet_uses_parameterized_query(
        self, sut_dml: Any, fake_conn: MagicMock
    ) -> None:
        dml_ops = sut_dml(fake_conn)
        dml_ops.insert_data("users", "gs://bucket/data.parquet")
        assert any(len(c[0]) > 1 for c in fake_conn.execute.call_args_list)

    def test_invalid_source_raises_typeerror(
        self, sut_dml: Any, fake_conn: MagicMock
    ) -> None:
        dml_ops = sut_dml(fake_conn)
        with pytest.raises(TypeError):
            dml_ops.insert_data("users", 12345)

    def test_invalid_name_raises_valueerror(
        self, sut_dml: Any, fake_conn: MagicMock
    ) -> None:
        dml_ops = sut_dml(fake_conn)
        with pytest.raises(ValueError, match="Invalid table name"):
            dml_ops.insert_data("BAD-name", sys.modules["pandas"].DataFrame())
