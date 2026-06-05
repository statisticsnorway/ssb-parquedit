"""Tests for ParquEdit - happy path and documented error behavior."""

import pandas as pd
import pytest

from ssb_parquedit.local import LocalDuckDBConnection
from ssb_parquedit.parquedit import ParquEdit

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def pe(conn: LocalDuckDBConnection) -> ParquEdit:
    """ParquEdit instance backed by a real local connection."""
    return ParquEdit.from_connection(conn)


# ── create_table: product_name validation ─────────────────────────────────────


class TestCreateTableProductNameRequired:
    """create_table() must enforce that product_name is provided and non-empty."""

    def test_raises_value_error_when_product_name_is_none(self, pe: ParquEdit) -> None:
        df = pd.DataFrame({"id": [1], "value": ["a"]})
        with pytest.raises(ValueError):
            pe.create_table("my_table", source=df, product_name=None, user_defined_id=["id"])

    def test_raises_value_error_when_product_name_is_empty_string(
        self, pe: ParquEdit
    ) -> None:
        df = pd.DataFrame({"id": [1], "value": ["a"]})
        with pytest.raises(ValueError):
            pe.create_table("my_table", source=df, product_name="")

    def test_error_message_is_informative(self, pe: ParquEdit) -> None:
        """The ValueError message should guide the user toward the fix."""
        df = pd.DataFrame({"id": [1], "value": ["a"]})
        with pytest.raises(ValueError, match="product_name"):
            pe.create_table("my_table", source=df, product_name=None)


# ── Happy path ────────────────────────────────────────────────────────────────


class TestParquEditHappyPath:
    """Core ParquEdit operations that must work under normal conditions."""

    def test_created_table_is_visible(self, pe: ParquEdit) -> None:
        df = pd.DataFrame({"id": [1], "name": ["Oslo"]})
        pe.create_table("cities", source=df, product_name="test", user_defined_id=["id"])
        assert pe.exists("cities")

    def test_create_with_fill_inserts_rows(self, pe: ParquEdit) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Oslo", "Bergen", "Tromsø"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )
        assert pe.count("cities") == 3

    def test_insert_data_adds_rows(self, pe: ParquEdit) -> None:
        df = pd.DataFrame({"id": [1, 2], "name": ["Oslo", "Bergen"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )
        pe.insert_data("cities", pd.DataFrame({"id": [3], "name": ["Tromsø"]}))
        assert pe.count("cities") == 3

    def test_view_returns_all_rows(self, pe: ParquEdit) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Oslo", "Bergen", "Tromsø"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )
        result = pe.view("cities")
        assert len(result) == 3

    def test_view_limit_is_respected(self, pe: ParquEdit) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Oslo", "Bergen", "Tromsø"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )
        result = pe.view("cities", limit=1)
        assert len(result) == 1

    def test_count_with_filter(self, pe: ParquEdit) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Oslo", "Bergen", "Tromsø"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )
        assert pe.count("cities", where="name='Oslo'") == 1

    def test_list_tables_includes_created_table(self, pe: ParquEdit) -> None:
        df = pd.DataFrame({"id": [1], "name": ["Oslo"]})
        pe.create_table("cities", source=df, product_name="test", user_defined_id=["id"])
        assert "cities" in pe.list_tables()

    def test_context_manager_closes_connection_on_exit(
        self, conn: LocalDuckDBConnection
    ) -> None:
        with ParquEdit.from_connection(conn) as pe_ctx:
            df = pd.DataFrame({"id": [1]})
            pe_ctx.create_table("t", source=df, product_name="test", user_defined_id=["id"])
        assert pe_ctx._conn is None
