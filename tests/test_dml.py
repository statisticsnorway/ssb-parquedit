"""Unit tests for DMLOperations.delete_row() and ParquEdit.delete_row()."""

import pandas as pd
import pytest

from ssb_parquedit.dml import DMLOperations
from ssb_parquedit.local import LocalDuckDBConnection
from ssb_parquedit.parquedit import ParquEdit

# ── Fixtures ──────────────────────────────────────────────────────────────────


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
    pe.create_table(
        "cities",
        source=df,
        product_name="test_product",
        user_defined_id=["id"],
        fill=True,
    )
    return pe


# ── delete_row: happy path ────────────────────────────────────────────────────


class TestDeleteRowHappyPath:
    def test_deletes_the_row(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        assert cities_table.count("cities") == 2

    def test_deleted_row_is_no_longer_visible(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        result = cities_table.view("cities")
        assert "Oslo" not in result["name"].tolist()

    def test_other_rows_are_unaffected(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        result = cities_table.view("cities")
        assert set(result["name"].tolist()) == {"Bergen", "Tromsø"}

    def test_accepts_numpy_int_rowid(self, cities_table: ParquEdit) -> None:
        """Rowid values pulled from a DataFrame column (e.g. numpy.int64) must work in `where`."""
        row = cities_table.view("cities", where="name = 'Oslo'")
        rowid = row["rowid"].iloc[0]
        cities_table.delete_row(
            "cities",
            where=f"rowid = {rowid}",
            change_event_reason="OTHER",
            change_comment="test",
        )
        assert cities_table.count("cities") == 2

    def test_deletes_multiple_matching_rows(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="population < 300000",
            change_event_reason="OTHER",
            change_comment="test",
        )
        result = cities_table.view("cities")
        assert result["name"].tolist() == ["Oslo"]

    def test_logs_one_changelog_entry_per_deleted_row(
        self, cities_table: ParquEdit
    ) -> None:
        cities_table.delete_row(
            "cities",
            where="population < 300000",
            change_event_reason="OTHER",
            change_comment="test",
        )
        edits = cities_table.get_edits("cities")
        assert len(edits) == 2
        assert set(edits["user_defined_id"].apply(lambda d: d["id"])) == {2, 3}


# ── delete_row: validation ────────────────────────────────────────────────────


class TestDeleteRowValidation:
    def test_raises_for_invalid_change_event_reason(
        self, cities_table: ParquEdit
    ) -> None:
        with pytest.raises(ValueError, match="Invalid cause"):
            cities_table.delete_row(
                "cities",
                where="rowid = 0",
                change_event_reason="NOT_A_REASON",
                change_comment="x",
            )

    def test_raises_for_unknown_table(self, cities_table: ParquEdit) -> None:
        with pytest.raises(TypeError, match="does not exist"):
            cities_table.delete_row(
                "no_such_table",
                where="rowid = 0",
                change_event_reason="OTHER",
                change_comment="x",
            )

    def test_raises_when_where_matches_no_rows(self, cities_table: ParquEdit) -> None:
        with pytest.raises(ValueError, match="No rows"):
            cities_table.delete_row(
                "cities",
                where="rowid = 999",
                change_event_reason="OTHER",
                change_comment="x",
            )

    def test_invalid_reason_does_not_delete_row(self, cities_table: ParquEdit) -> None:
        with pytest.raises(ValueError):
            cities_table.delete_row(
                "cities",
                where="rowid = 0",
                change_event_reason="NOT_A_REASON",
                change_comment="x",
            )
        assert cities_table.count("cities") == 3


# ── delete_row: changelog metadata ────────────────────────────────────────────


class TestDeleteRowChangelog:
    """delete_row() must log the deletion as an edit, like edit() does."""

    def test_get_edits_includes_the_deletion(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="REVIEW",
            change_comment="removed duplicate",
        )
        edits = cities_table.get_edits("cities")
        assert len(edits) == 1
        assert edits["change_event_reason"].iloc[0] == "REVIEW"
        assert edits["change_comment"].iloc[0] == "removed duplicate"

    def test_new_values_is_none(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        edits = cities_table.get_edits("cities")
        assert edits["new_values"].iloc[0] is None

    def test_old_values_contains_full_row(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        edits = cities_table.get_edits("cities")
        old_values = edits["old_values"].iloc[0]
        assert old_values == {"id": 1, "name": "Oslo", "population": 700000}

    def test_change_type_is_delete(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        edits = cities_table.get_edits("cities")
        assert edits["change_type"].iloc[0] == "DELETE"

    def test_user_defined_id_is_recorded(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        edits = cities_table.get_edits("cities")
        assert edits["user_defined_id"].iloc[0] == {"id": 1}

    def test_edit_still_reports_change_type_update(
        self, cities_table: ParquEdit
    ) -> None:
        """edit() must keep tagging its own changelog entries as 'UPDATE'."""
        cities_table.edit(
            "cities",
            rowid=0,
            changes={"name": "Oslo edited"},
            change_event_reason="OTHER",
            change_comment="test",
        )
        edits = cities_table.get_edits("cities")
        assert edits["change_type"].iloc[0] == "UPDATE"


# ── DMLOperations.delete_row: direct unit tests ───────────────────────────────


class TestDMLOperationsDeleteRow:
    def test_no_op_when_table_has_no_tag_info(
        self, conn: LocalDuckDBConnection
    ) -> None:
        """Mirrors edit()'s behavior: silently returns if the table has no product/user_defined_id tag."""
        df = pd.DataFrame({"id": [1], "name": ["Oslo"]})
        conn.register("data", df)
        conn.execute("CREATE TABLE cities AS SELECT * FROM data")

        dml = DMLOperations(conn)
        dml.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        # Row must still be present - deletion never happened.
        assert conn.execute("SELECT COUNT(*) FROM cities").fetchone()[0] == 1

    def test_rollback_on_failure_leaves_row_intact(
        self, cities_table: ParquEdit
    ) -> None:
        """If set_commit_message fails mid-transaction, the DELETE must be rolled back.

        Uses a ValueError so tenacity's retry (which skips ValueError/TypeError)
        doesn't retry this for several seconds before giving up.
        """
        conn = cities_table._get_connection()
        dml = DMLOperations(conn, cities_table._db_config)

        original_execute = conn.execute

        def failing_execute(sql: str, parameters: list[object] | None = None) -> object:
            if isinstance(sql, str) and "set_commit_message" in sql:
                raise ValueError("boom")
            return original_execute(sql, parameters)

        conn.execute = failing_execute  # type: ignore[method-assign]
        try:
            with pytest.raises(ValueError, match="boom"):
                dml.delete_row(
                    "cities",
                    where="rowid = 0",
                    change_event_reason="OTHER",
                    change_comment="test",
                )
        finally:
            conn.execute = original_execute  # type: ignore[method-assign]

        assert cities_table.count("cities") == 3
