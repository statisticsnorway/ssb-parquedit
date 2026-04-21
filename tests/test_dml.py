"""Tests for DML (Data Manipulation Language) operations."""

from typing import Any
from unittest.mock import MagicMock

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_dml() -> Any:
    """Import and return the DMLOperations class."""
    import importlib

    module = importlib.import_module("ssb_parquedit.dml")
    importlib.reload(module)
    return module.DMLOperations


class TestInsertDataTypeErrors:
    """Test error handling for invalid source types."""

    def test_insert_with_invalid_source_type(
        self, sut_dml: Any, fake_conn: MagicMock
    ) -> None:
        """Test that invalid source type raises TypeError."""
        dml_ops: Any = sut_dml(fake_conn)

        with pytest.raises(TypeError, match="source must be a DataFrame"):
            dml_ops.insert_data("users", 12345)
