# tests/integration/test_integration_local.py
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ssb_parquedit.parquedit import ParquEdit


@pytest.fixture
def byer_tabell(pe: ParquEdit) -> ParquEdit:
    """Hjelpefixture - oppretter og populerer en standardtabell."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "navn": ["Oslo", "Bergen", "Tromsø"],
            "innbyggere": [700000, 285000, 75000],
        }
    )
    pe.create_table("byer", source=df, product_name="test_produkt", fill=True)
    return pe


# ============ create / exists / count ============


def test_create_table_exists(pe: ParquEdit) -> None:
    df = pd.DataFrame({"id": [1], "navn": ["Oslo"]})
    pe.create_table("byer", source=df, product_name="test_produkt")
    assert pe.exists("byer")


def test_create_and_count(byer_tabell: ParquEdit) -> None:
    assert byer_tabell.count("byer") == 3


def test_count_where(byer_tabell: ParquEdit) -> None:
    assert byer_tabell.count("byer", where="navn='Oslo'") == 1


def test_list_tables(byer_tabell: ParquEdit) -> None:
    assert "byer" in byer_tabell.list_tables()


# ============ view - grunnleggende ============


def test_view_returns_all_rows(byer_tabell: ParquEdit) -> None:
    result = byer_tabell.view("byer")
    assert len(result) == 3


def test_view_default_includes_rowid(byer_tabell: ParquEdit) -> None:
    result = byer_tabell.view("byer")
    assert "rowid" in result.columns


# ============ view - where ============


def test_view_where_filters_rows(byer_tabell: ParquEdit) -> None:
    result = byer_tabell.view("byer", where="id > 1")
    assert len(result) == 2
    assert "Oslo" not in result["navn"].values


def test_view_where_no_match_returns_empty(byer_tabell: ParquEdit) -> None:
    result = byer_tabell.view("byer", where="id = 999")
    assert len(result) == 0


def test_view_where_none_returns_all(byer_tabell: ParquEdit) -> None:
    result = byer_tabell.view("byer", where=None)
    assert len(result) == 3


def test_view_where_string_comparison(byer_tabell: ParquEdit) -> None:
    result = byer_tabell.view("byer", where="navn = 'Oslo'")
    assert len(result) == 1
    assert result["navn"].iloc[0] == "Oslo"


def test_view_where_compound_condition(byer_tabell: ParquEdit) -> None:
    result = byer_tabell.view("byer", where="id > 1 AND innbyggere > 100000")
    assert len(result) == 1
    assert result["navn"].iloc[0] == "Bergen"


# ============ view - kombinasjoner ============


def test_view_where_with_limit(byer_tabell: ParquEdit) -> None:
    result = byer_tabell.view("byer", where="id > 0", limit=2)
    assert len(result) == 2


def test_view_where_with_order_by(byer_tabell: ParquEdit) -> None:
    result = byer_tabell.view("byer", where="id > 0", order_by="innbyggere ASC")
    assert result["navn"].iloc[0] == "Tromsø"


def test_view_where_with_offset(byer_tabell: ParquEdit) -> None:
    result = byer_tabell.view("byer", where="id > 0", order_by="id ASC", offset=1)
    assert result["navn"].iloc[0] == "Bergen"


# ============ view- columns-bug ============


def test_view_columns_subset_includes_rowid(byer_tabell: ParquEdit) -> None:
    """Gjenskaper select_clause-buggen: 'rowid, '.join(columns) er feil."""
    result = byer_tabell.view("byer", columns=["id", "navn"])
    assert "rowid" in result.columns
    assert "id" in result.columns
    assert "navn" in result.columns
    assert "innbyggere" not in result.columns


# ============ insert ============


def test_insert_data(pe: ParquEdit) -> None:
    df = pd.DataFrame({"id": [1], "navn": ["Oslo"]})
    pe.create_table("byer", source=df, product_name="test_produkt")
    df2 = pd.DataFrame({"id": [2], "navn": ["Bergen"]})
    pe.insert_data("byer", df2)
    assert pe.count("byer") == 1  # første insert skjedde ikke (fill=False)


def test_insert_from_parquet(pe: ParquEdit, tmp_storage: str) -> None:
    """Tester insert_data fra en lokal Parquet-fil."""
    parquet_path = str(Path(tmp_storage) / "byer.parquet")
    tabell = pa.table({"id": [3, 4], "navn": ["Tromsø", "Stavanger"]})
    pq.write_table(tabell, parquet_path)

    df = pd.DataFrame({"id": [1, 2], "navn": ["Oslo", "Bergen"]})
    pe.create_table("byer", source=df, product_name="test_produkt")
    pe.insert_data("byer", parquet_path)

    assert pe.count("byer") == 2


def test_create_table_fill_from_parquet(pe: ParquEdit, tmp_storage: str) -> None:
    """Tester create_table med fill=True fra en lokal Parquet-fil."""
    parquet_path = str(Path(tmp_storage) / "byer.parquet")
    tabell = pa.table({"id": [1, 2, 3], "navn": ["Oslo", "Bergen", "Tromsø"]})
    pq.write_table(tabell, parquet_path)

    pe.create_table("byer", source=parquet_path, product_name="test_produkt", fill=True)

    assert pe.count("byer") == 3


# ============ drop ============


def test_drop_table(byer_tabell: ParquEdit) -> None:
    byer_tabell.drop_table("byer", cleanup=False)
    assert not byer_tabell.exists("byer")


# ============ rowid ============


def test_view_returns_rowid(byer_tabell: ParquEdit) -> None:
    """view() skal alltid inkludere rowid-kolonne."""
    result = byer_tabell.view("byer")
    assert "rowid" in result.columns


def test_rowid_is_unique(byer_tabell: ParquEdit) -> None:
    """Hver rad skal ha unik rowid."""
    result = byer_tabell.view("byer")
    assert result["rowid"].nunique() == len(result)


def test_rowid_is_integer(byer_tabell: ParquEdit) -> None:
    """Rowid skal være av heltallstype."""
    result = byer_tabell.view("byer")
    assert pd.api.types.is_integer_dtype(result["rowid"])


def test_rowid_usable_in_where(byer_tabell: ParquEdit) -> None:
    """Rowid skal kunne brukes som filter i where-parameteren."""
    alle = byer_tabell.view("byer")
    første_rowid = alle["rowid"].iloc[0]

    result = byer_tabell.view("byer", where=f"rowid = {første_rowid}")
    assert len(result) == 1
    assert result["rowid"].iloc[0] == første_rowid


def test_edit_via_rowid(byer_tabell: ParquEdit) -> None:
    """edit() skal oppdatere korrekt rad via rowid."""
    alle = byer_tabell.view("byer")
    forste_rowid = int(alle["rowid"].iloc[0])

    byer_tabell.edit(
        "byer",
        rowid=forste_rowid,
        changes={"navn": "Kristiansand"},
        change_event_reason="OTHER",
        change_comment="Testendring",
    )

    result = byer_tabell.view("byer", where=f"rowid = {forste_rowid}")
    assert result["navn"].iloc[0] == "Kristiansand"
