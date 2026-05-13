# tests/integration/test_integration_local.py
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ssb_parquedit.parquedit import ParquEdit


@pytest.fixture
def cities_table(pe: ParquEdit) -> ParquEdit:
    """Helper fixture - creates and populates a standard table."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Oslo", "Bergen", "Tromsø"],
            "population": [700000, 285000, 75000],
        }
    )
    pe.create_table("cities", source=df, product_name="test_product", fill=True)
    return pe


# ============ create / exists / count ============


def test_create_table_exists(pe: ParquEdit) -> None:
    df = pd.DataFrame({"id": [1], "name": ["Oslo"]})
    pe.create_table("cities", source=df, product_name="test_product")
    assert pe.exists("cities")


def test_create_and_count(cities_table: ParquEdit) -> None:
    assert cities_table.count("cities") == 3


def test_count_where(cities_table: ParquEdit) -> None:
    assert cities_table.count("cities", where="name='Oslo'") == 1


def test_list_tables(cities_table: ParquEdit) -> None:
    assert "cities" in cities_table.list_tables()


# ============ view - basic ============


def test_view_returns_all_rows(cities_table: ParquEdit) -> None:
    result = cities_table.view("cities")
    assert len(result) == 3


def test_view_default_includes_rowid(cities_table: ParquEdit) -> None:
    result = cities_table.view("cities")
    assert "rowid" in result.columns


# ============ view - where ============


def test_view_where_filters_rows(cities_table: ParquEdit) -> None:
    result = cities_table.view("cities", where="id > 1")
    assert len(result) == 2
    assert "Oslo" not in result["name"].values


def test_view_where_no_match_returns_empty(cities_table: ParquEdit) -> None:
    result = cities_table.view("cities", where="id = 999")
    assert len(result) == 0


def test_view_where_none_returns_all(cities_table: ParquEdit) -> None:
    result = cities_table.view("cities", where=None)
    assert len(result) == 3


def test_view_where_string_comparison(cities_table: ParquEdit) -> None:
    result = cities_table.view("cities", where="name = 'Oslo'")
    assert len(result) == 1
    assert result["name"].iloc[0] == "Oslo"


def test_view_where_compound_condition(cities_table: ParquEdit) -> None:
    result = cities_table.view("cities", where="id > 1 AND population > 100000")
    assert len(result) == 1
    assert result["name"].iloc[0] == "Bergen"


# ============ view - combinations ============


def test_view_where_with_limit(cities_table: ParquEdit) -> None:
    result = cities_table.view("cities", where="id > 0", limit=2)
    assert len(result) == 2


def test_view_where_with_order_by(cities_table: ParquEdit) -> None:
    result = cities_table.view("cities", where="id > 0", order_by="population ASC")
    assert result["name"].iloc[0] == "Tromsø"


def test_view_where_with_offset(cities_table: ParquEdit) -> None:
    result = cities_table.view("cities", where="id > 0", order_by="id ASC", offset=1)
    assert result["name"].iloc[0] == "Bergen"


# ============ view - columns bug ============


def test_view_columns_subset_includes_rowid(cities_table: ParquEdit) -> None:
    """Reproduces the select_clause bug: 'rowid, '.join(columns) is wrong."""
    result = cities_table.view("cities", columns=["id", "name"])
    assert "rowid" in result.columns
    assert "id" in result.columns
    assert "name" in result.columns
    assert "population" not in result.columns


# ============ insert ============


def test_insert_data(pe: ParquEdit) -> None:
    df = pd.DataFrame({"id": [1], "name": ["Oslo"]})
    pe.create_table("cities", source=df, product_name="test_product")
    df2 = pd.DataFrame({"id": [2], "name": ["Bergen"]})
    pe.insert_data("cities", df2)
    assert pe.count("cities") == 1  # first insert did not happen (fill=False)


def test_insert_from_parquet(pe: ParquEdit, tmp_storage: str) -> None:
    """Tests insert_data from a local Parquet file."""
    parquet_path = str(Path(tmp_storage) / "cities.parquet")
    table = pa.table({"id": [3, 4], "name": ["Tromsø", "Stavanger"]})
    pq.write_table(table, parquet_path)

    df = pd.DataFrame({"id": [1, 2], "name": ["Oslo", "Bergen"]})
    pe.create_table("cities", source=df, product_name="test_product")
    pe.insert_data("cities", parquet_path)

    assert pe.count("cities") == 2


def test_create_table_fill_from_parquet(pe: ParquEdit, tmp_storage: str) -> None:
    """Tests create_table with fill=True from a local Parquet file."""
    parquet_path = str(Path(tmp_storage) / "cities.parquet")
    table = pa.table({"id": [1, 2, 3], "name": ["Oslo", "Bergen", "Tromsø"]})
    pq.write_table(table, parquet_path)

    pe.create_table(
        "cities", source=parquet_path, product_name="test_product", fill=True
    )

    assert pe.count("cities") == 3


# ============ drop ============


def test_drop_table(cities_table: ParquEdit) -> None:
    cities_table.drop_table("cities", cleanup=False)
    assert not cities_table.exists("cities")


# ============ rowid ============


def test_view_returns_rowid(cities_table: ParquEdit) -> None:
    """view() should always include rowid column."""
    result = cities_table.view("cities")
    assert "rowid" in result.columns


def test_rowid_is_unique(cities_table: ParquEdit) -> None:
    """Each row should have a unique rowid."""
    result = cities_table.view("cities")
    assert result["rowid"].nunique() == len(result)


def test_rowid_is_integer(cities_table: ParquEdit) -> None:
    """Rowid should be of integer type."""
    result = cities_table.view("cities")
    assert pd.api.types.is_integer_dtype(result["rowid"])


def test_rowid_usable_in_where(cities_table: ParquEdit) -> None:
    """Rowid should be usable as a filter in the where parameter."""
    all_rows = cities_table.view("cities")
    first_rowid = all_rows["rowid"].iloc[0]

    result = cities_table.view("cities", where=f"rowid = {first_rowid}")
    assert len(result) == 1
    assert result["rowid"].iloc[0] == first_rowid


def test_edit_via_rowid(cities_table: ParquEdit) -> None:
    """edit() should update the correct row via rowid."""
    all_rows = cities_table.view("cities")
    first_rowid = int(all_rows["rowid"].iloc[0])

    cities_table.edit(
        "cities",
        rowid=first_rowid,
        changes={"name": "Kristiansand"},
        change_event_reason="OTHER",
        change_comment="Test edit",
    )

    result = cities_table.view("cities", where=f"rowid = {first_rowid}")
    assert result["name"].iloc[0] == "Kristiansand"
