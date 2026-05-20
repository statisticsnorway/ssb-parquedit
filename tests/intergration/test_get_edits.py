import pandas as pd
import pytest

from ssb_parquedit.parquedit import ParquEdit


@pytest.fixture
def edited_cities(pe: ParquEdit) -> ParquEdit:
    """Creates a cities table, inserts data, and performs an edit to generate snapshot history."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Oslo", "Bergen", "Tromsø"],
            "population": [700000, 285000, 75000],
        }
    )
    pe.create_table("cities", source=df, product_name="test_product", fill=True)

    all_rows = pe.view("cities")
    first_rowid = int(all_rows["rowid"].iloc[0])

    pe.edit(
        "cities",
        rowid=first_rowid,
        changes={"name": "Kristiansand"},
        change_event_reason="OTHER",
        change_comment="Rename Oslo to Kristiansand",
    )
    return pe


# ============ get_edits - basic ============


def test_get_edits_returns_result(edited_cities: ParquEdit) -> None:
    result = edited_cities.get_edits("cities").df()
    assert result is not None


def test_get_edits_not_empty(edited_cities: ParquEdit) -> None:
    result = edited_cities.get_edits("cities").df()
    assert len(result) > 0


def test_get_edits_expected_columns(edited_cities: ParquEdit) -> None:
    result = edited_cities.get_edits("cities").df()
    expected_cols = {
        "snapshot_id",
        "rowid",
        "snapshot_time",
        "author",
        "commit_message",
        "commit_extra_info",
        "var",
        "value",
        "pre_value",
    }
    assert expected_cols.issubset(set(result.columns))


# ============ get_edits - content ============


def test_get_edits_detects_changed_column(edited_cities: ParquEdit) -> None:
    result = edited_cities.get_edits("cities").df()
    assert "name" in result["var"].values


def test_get_edits_correct_new_value(edited_cities: ParquEdit) -> None:
    result = edited_cities.get_edits("cities").df()
    name_edits = result[result["var"] == "name"]
    assert "Kristiansand" in name_edits["value"].values


def test_get_edits_correct_pre_value(edited_cities: ParquEdit) -> None:
    result = edited_cities.get_edits("cities").df()
    name_edits = result[result["var"] == "name"]
    assert "Oslo" in name_edits["pre_value"].values


def test_get_edits_unchanged_columns_not_in_result(edited_cities: ParquEdit) -> None:
    """Columns that did not change should not appear as edits for the edited row."""
    result = edited_cities.get_edits("cities").df()
    all_rows = edited_cities.view("cities")
    first_rowid = int(all_rows["rowid"].iloc[0])

    row_edits = result[result["rowid"] == first_rowid]
    # id and population were not changed - should not appear
    assert "id" not in row_edits["var"].values
    assert "population" not in row_edits["var"].values


def test_get_edits_only_edited_row_appears(edited_cities: ParquEdit) -> None:
    """Only the rowid that was edited should appear in the result."""
    result = edited_cities.get_edits("cities").df()
    all_rows = edited_cities.view("cities")
    first_rowid = int(all_rows["rowid"].iloc[0])

    assert set(result["rowid"].unique()) == {first_rowid}


# ============ get_edits - no edits case ============


def test_get_edits_no_edits_returns_empty(pe: ParquEdit) -> None:
    """A table with only inserts (no updates) should return no edits."""
    df = pd.DataFrame({"id": [1, 2], "name": ["Oslo", "Bergen"]})
    pe.create_table("cities", source=df, product_name="test_product", fill=True)

    result = pe.get_edits("cities").df()
    assert len(result) == 0


# ============ get_edits - multiple edits ============


def test_get_edits_multiple_edits_tracked(pe: ParquEdit) -> None:
    """Multiple sequential edits on the same row should each appear."""
    df = pd.DataFrame({"id": [1], "name": ["Oslo"], "population": [700000]})
    pe.create_table("cities", source=df, product_name="test_product", fill=True)

    all_rows = pe.view("cities")
    rowid = int(all_rows["rowid"].iloc[0])

    pe.edit(
        "cities",
        rowid=rowid,
        changes={"name": "Kristiansand"},
        change_event_reason="OTHER",
        change_comment="First edit",
    )
    pe.edit(
        "cities",
        rowid=rowid,
        changes={"name": "Stavanger"},
        change_event_reason="OTHER",
        change_comment="Second edit",
    )

    result = pe.get_edits("cities").df()
    name_edits = result[result["var"] == "name"].sort_values("snapshot_id")
    assert len(name_edits) == 2
    assert name_edits["value"].iloc[-1] == "Stavanger"
