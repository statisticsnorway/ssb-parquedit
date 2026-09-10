"""Tests for ParquEdit - happy path and documented error behavior."""

import pandas as pd
import polars as pl
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
            pe.create_table(
                "my_table", source=df, product_name=None, user_defined_id=["id"]
            )

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
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"]
        )
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
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"]
        )
        assert "cities" in pe.list_tables()

    def test_context_manager_closes_connection_on_exit(
        self, conn: LocalDuckDBConnection
    ) -> None:
        with ParquEdit.from_connection(conn) as pe_ctx:
            df = pd.DataFrame({"id": [1]})
            pe_ctx.create_table(
                "t", source=df, product_name="test", user_defined_id=["id"]
            )
        assert pe_ctx._conn is None


class TestParquEditLocal:
    def test_local_can_flush_inlined_table(self, tmp_storage: str) -> None:
        pe = ParquEdit.local(tmp_storage)
        df = pd.DataFrame({"id": [1, 2], "name": ["Oslo", "Bergen"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )

        pe.flush_inlined_table("cities")

        assert pe.count("cities") == 2
        pe.close()


# ── Happy path: polars sources ────────────────────────────────────────────────


class TestParquEditHappyPathPolars:
    """create_table() and insert_data() must accept polars DataFrames too."""

    def test_created_table_is_visible(self, pe: ParquEdit) -> None:
        df = pl.DataFrame({"id": [1], "name": ["Oslo"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"]
        )
        assert pe.exists("cities")

    def test_create_with_fill_inserts_rows(self, pe: ParquEdit) -> None:
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["Oslo", "Bergen", "Tromsø"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )
        assert pe.count("cities") == 3

    def test_insert_data_adds_rows(self, pe: ParquEdit) -> None:
        df = pl.DataFrame({"id": [1, 2], "name": ["Oslo", "Bergen"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )
        pe.insert_data("cities", pl.DataFrame({"id": [3], "name": ["Tromsø"]}))
        assert pe.count("cities") == 3

    def test_insert_polars_into_pandas_created_table(self, pe: ParquEdit) -> None:
        """Table schema from pandas, data inserted from polars — cross-source support."""
        df = pd.DataFrame({"id": [1, 2], "name": ["Oslo", "Bergen"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )
        pe.insert_data("cities", pl.DataFrame({"id": [3], "name": ["Tromsø"]}))
        assert pe.count("cities") == 3

    def test_insert_pandas_into_polars_created_table(self, pe: ParquEdit) -> None:
        """Table schema from polars, data inserted from pandas — cross-source support."""
        df = pl.DataFrame({"id": [1, 2], "name": ["Oslo", "Bergen"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )
        pe.insert_data("cities", pd.DataFrame({"id": [3], "name": ["Tromsø"]}))
        assert pe.count("cities") == 3

    def test_insert_coerces_polars_int_column_to_bigint_target(
        self, pe: ParquEdit
    ) -> None:
        """A narrower polars integer dtype must still fit the table's BIGINT column."""
        df = pd.DataFrame({"id": [1], "count": pd.array([1], dtype="Int64")})
        pe.create_table(
            "counts", source=df, product_name="test", user_defined_id=["id"]
        )
        narrow_df = pl.DataFrame({"id": [2], "count": pl.Series([2], dtype=pl.Int32)})
        pe.insert_data("counts", narrow_df)
        result = pe.view("counts", where="id = 2")
        assert result["count"].iloc[0] == 2

    def test_insert_coerces_polars_numeric_column_to_varchar_target(
        self, pe: ParquEdit
    ) -> None:
        """A polars numeric column must be cast to string for a VARCHAR target column."""
        df = pd.DataFrame({"id": [1], "code": ["001"]})
        pe.create_table("codes", source=df, product_name="test", user_defined_id=["id"])
        numeric_df = pl.DataFrame({"id": [2], "code": [2]})
        pe.insert_data("codes", numeric_df)
        result = pe.view("codes", where="id = 2")
        assert result["code"].iloc[0] == "2"

    def test_view_returns_all_rows(self, pe: ParquEdit) -> None:
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["Oslo", "Bergen", "Tromsø"]})
        pe.create_table(
            "cities", source=df, product_name="test", user_defined_id=["id"], fill=True
        )
        result = pe.view("cities")
        assert len(result) == 3
