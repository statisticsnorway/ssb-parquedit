"""Tests for the ParquEdit facade."""

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, sut, db_config


class TestInit:
    def test_stores_provided_config(self, sut: Any, db_config: dict[str, str]) -> None:
        pe = sut(config=db_config)
        assert pe._db_config == db_config


class TestContextManager:
    def test_opens_and_closes_connection(
        self, sut: Any, db_config: dict[str, str]
    ) -> None:
        mock_conn = MagicMock()
        pe = sut(config=db_config)
        with patch("ssb_parquedit.parquedit.DuckDBConnection", return_value=mock_conn):
            with pe:
                assert pe._conn is not None
        assert pe._conn is None


class TestCreateTable:
    def test_raises_without_product_name(
        self, sut: Any, db_config: dict[str, str]
    ) -> None:
        pe = sut(config=db_config)
        with pytest.raises(ValueError, match="product_name"):
            pe.create_table("users", {}, product_name=None)

    def test_delegates_to_ddl(self, sut: Any, db_config: dict[str, str]) -> None:
        mock_conn = MagicMock()
        mock_ddl = MagicMock()
        pe = sut(config=db_config)
        with patch("ssb_parquedit.parquedit.DuckDBConnection", return_value=mock_conn):
            with patch("ssb_parquedit.parquedit.DDLOperations", return_value=mock_ddl):
                pe.create_table("users", {}, product_name="myproduct")
        mock_ddl.create_table.assert_called_once()


class TestDropTable:
    def test_delegates_to_ddl(self, sut: Any, db_config: dict[str, str]) -> None:
        mock_conn = MagicMock()
        mock_ddl = MagicMock()
        pe = sut(config=db_config)
        with patch("ssb_parquedit.parquedit.DuckDBConnection", return_value=mock_conn):
            with patch("ssb_parquedit.parquedit.DDLOperations", return_value=mock_ddl):
                pe.drop_table("users", cleanup=False)
        mock_ddl.drop_table.assert_called_once_with("users", cleanup=False)


class TestInsertData:
    def test_delegates_to_dml(self, sut: Any, db_config: dict[str, str]) -> None:
        mock_conn = MagicMock()
        mock_dml = MagicMock()
        pe = sut(config=db_config)
        with patch("ssb_parquedit.parquedit.DuckDBConnection", return_value=mock_conn):
            with patch("ssb_parquedit.parquedit.DMLOperations", return_value=mock_dml):
                pe.insert_data("users", "gs://bucket/data.parquet")
        mock_dml.insert_data.assert_called_once()


class TestView:
    def test_delegates_to_query(self, sut: Any, db_config: dict[str, str]) -> None:
        mock_conn = MagicMock()
        mock_query = MagicMock()
        pe = sut(config=db_config)
        with patch("ssb_parquedit.parquedit.DuckDBConnection", return_value=mock_conn):
            with patch(
                "ssb_parquedit.parquedit.QueryOperations", return_value=mock_query
            ):
                pe.view("users")
        mock_query.view.assert_called_once()
