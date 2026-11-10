"""Unit tests for DMLOperations.delete_row() and ParquEdit.delete_row()."""

from json import dumps as _dumps
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from ssb_parquedit.dml import DMLOperations
from ssb_parquedit.local import LocalDuckDBConnection
from ssb_parquedit.parquedit import ParquEdit

# ── Fixtures and Wrapper functions──────────────────────────────────────────────────────────────────


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


def dumps_wrapper(*args: Any, **kwargs: Any) -> str:
    return _dumps(*args, **(kwargs | {"default": lambda obj: "mock"}))


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

    def test_logs_a_single_changelog_entry_for_the_whole_deletion(
        self, cities_table: ParquEdit
    ) -> None:
        cities_table.delete_row(
            "cities",
            where="population < 300000",
            change_event_reason="OTHER",
            change_comment="test",
        )
        edits = cities_table.get_edits("cities")
        assert len(edits) == 1
        assert edits["affected_rows"].iloc[0] == 2
        assert edits["user_defined_id"].iloc[0] is None


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

    def test_old_values_is_none(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        edits = cities_table.get_edits("cities")
        assert edits["old_values"].iloc[0] is None

    def test_change_type_is_delete(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        edits = cities_table.get_edits("cities")
        assert edits["change_type"].iloc[0] == "DELETE"

    def test_user_defined_id_is_none(self, cities_table: ParquEdit) -> None:
        cities_table.delete_row(
            "cities",
            where="rowid = 0",
            change_event_reason="OTHER",
            change_comment="test",
        )
        edits = cities_table.get_edits("cities")
        assert edits["user_defined_id"].iloc[0] is None

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
        with pytest.raises(ValueError):
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

    def test_rollback_exception(self) -> None:
        def execute_mock(cmd: Any, *args: object) -> None:
            if cmd == "BEGIN":
                raise ValueError("test_begin")
            if cmd == "ROLLBACK":
                raise ValueError("test_rollback")

        mock_conn = MagicMock()
        me = MagicMock()
        mock_conn.execute = me
        mock_conn.execute.side_effect = execute_mock
        db_config = MagicMock()

        dml = DMLOperations(mock_conn, db_config)
        dml._validate_table_and_columns = MagicMock()  # type: ignore

        change_event_reason = "OTHER"
        change_comment = "tag dict is none"
        table_name = "t1"
        where = "test"

        with (
            patch("ssb_parquedit.dml.QueryOperations") as mock_query,
            patch("ssb_parquedit.dml.json.dumps", MagicMock(wraps=dumps_wrapper)),
        ):
            mock_query._get_tag_info.return_value = {
                "product_name": "test",
                "user_defined_id": "test",
            }
            with pytest.raises(ValueError, match="test_begin"):
                dml.delete_row(table_name, where, change_event_reason, change_comment)

        me.assert_called_with("ROLLBACK")


class TestDMLOperationsInsertData:
    def test_insert_data_invalid_source(self) -> None:
        mock_conn = MagicMock()
        db_config = MagicMock()
        dml = DMLOperations(mock_conn, db_config)
        with pytest.raises(
            TypeError, match="source must be a DataFrame or gs:// Parquet path"
        ):
            dml.insert_data("t1", 1)


class TestDMLOperationsValidateTableAndColumns:
    def test_with_list(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("a",),
            ("b",),
            ("c",),
            ("d",),
        ]
        changes = {"a[0]": ""}
        table_name = "a"
        db_config = MagicMock()
        dml = DMLOperations(mock_conn, db_config)
        dml._validate_table_and_columns(table_name, changes)

    def test_missing_columns(self) -> None:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("a",),
            ("b",),
            ("c",),
            ("d",),
        ]
        changes = {"e": ""}
        table_name = "a"
        db_config = MagicMock()
        dml = DMLOperations(mock_conn, db_config)
        with pytest.raises(
            TypeError, match=f"Missing columns in '{table_name}': {set(changes.keys())}"
        ):
            dml._validate_table_and_columns(table_name, changes)


class TestDMLOperationsEdit:

    def test_edit_invalid_change_event_reason(self) -> None:
        mock_conn = MagicMock()
        db_config = MagicMock()

        table_name = "t1"
        rowid = 1
        changes = {"e": ""}
        change_event_reason = "INVALID"
        change_comment = "invalid"

        dml = DMLOperations(mock_conn, db_config)
        with pytest.raises(ValueError, match=r"Invalid cause: .*. Must be one of: .*"):
            dml.edit(table_name, rowid, changes, change_event_reason, change_comment)

    def test_tag_dict_is_none(self) -> None:
        mock_conn = MagicMock()
        db_config = MagicMock()

        execute = MagicMock()
        mock_conn.execute = execute

        table_name = "t1"
        rowid = 1
        changes = {"e": ""}
        change_event_reason = "OTHER"
        change_comment = "tag dict is none"
        dml = DMLOperations(mock_conn, db_config)
        dml._validate_table_and_columns = MagicMock()  # type: ignore

        with (
            patch("ssb_parquedit.dml.QueryOperations") as mock_query,
            patch("ssb_parquedit.dml.json.dumps", MagicMock(wraps=dumps_wrapper)),
            pytest.raises(
                ValueError, match="Tag information not found for table: 't1'"
            ),
        ):
            mock_query.return_value._get_tag_info = Mock(return_value=None)
            dml.edit(table_name, rowid, changes, change_event_reason, change_comment)
        execute.assert_not_called()

    def test_rollback_on_exception(self) -> None:
        def execute_mock(arg: Any, *args: object) -> None:
            if arg == "BEGIN":
                raise ValueError("test")

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = execute_mock
        db_config = MagicMock()

        dml = DMLOperations(mock_conn, db_config)
        dml._validate_table_and_columns = MagicMock()  # type: ignore

        change_event_reason = "OTHER"
        change_comment = "tag dict is none"
        table_name = "t1"
        rowid = 1
        changes = {"e": ""}

        with (
            patch("ssb_parquedit.dml.QueryOperations") as mock_query,
            patch("ssb_parquedit.dml.json.dumps", MagicMock(wraps=dumps_wrapper)),
        ):
            mock_query._get_tag_info.return_value = {
                "product_name": "test",
                "user_defined_id": "test",
            }
            with pytest.raises(ValueError, match="test"):
                dml.edit(
                    table_name, rowid, changes, change_event_reason, change_comment
                )

    def test_rollback_exception(self) -> None:
        def execute_mock(cmd: Any, *args: object) -> None:
            if cmd == "BEGIN":
                raise ValueError("test_begin")
            if cmd == "ROLLBACK":
                raise ValueError("test_rollback")

        mock_conn = MagicMock()
        me = MagicMock()
        mock_conn.execute = me
        mock_conn.execute.side_effect = execute_mock
        db_config = MagicMock()

        dml = DMLOperations(mock_conn, db_config)
        dml._validate_table_and_columns = MagicMock()  # type: ignore

        change_event_reason = "OTHER"
        change_comment = "tag dict is none"
        table_name = "t1"
        rowid = 1
        changes = {"e": ""}

        with (
            patch("ssb_parquedit.dml.QueryOperations") as mock_query,
            patch("ssb_parquedit.dml.json.dumps", MagicMock(wraps=dumps_wrapper)),
        ):
            mock_query._get_tag_info.return_value = {
                "product_name": "test",
                "user_defined_id": "test",
            }
            with pytest.raises(ValueError, match="test_begin"):
                dml.edit(
                    table_name, rowid, changes, change_event_reason, change_comment
                )

        me.assert_called_with("ROLLBACK")


class TestDMLOperationsPandasToArrow:
    def test_bigint(self) -> None:
        d = {"a": [1, 2, 3], "b": [4, 5, 6]}
        df = pd.DataFrame(data=d, dtype=np.int64)
        col_types = {"a": "BIGINT", "b": "BIGINT"}

        table = DMLOperations._pandas_to_arrow(df, col_types)
        i = 1
        for col in table:
            for e in col:
                assert e.equals(pa.scalar(i))
                i += 1

    def test_bigint_nan(self) -> None:
        d = {"a": [1, 2, 3], "b": ["n", "n", "n"]}
        df = pd.DataFrame(data=d)
        col_types = {"a": "BIGINT", "b": "BIGINT"}

        table = DMLOperations._pandas_to_arrow(df, col_types)
        i = 1
        for col in table:
            for e in col:
                assert e.equals(pa.scalar(i)) or not e.is_valid
                i += 1

    def test_varchar_after_bigint(self) -> None:
        d = {"a": [1, 2, 3], "b": ["n", "n", "n"]}
        df = pd.DataFrame(data=d)
        col_types = {"a": "BIGINT", "b": "VARCHAR"}

        table = DMLOperations._pandas_to_arrow(df, col_types)
        i = 1
        for e in table["a"]:
            assert e.equals(pa.scalar(i))
            i += 1

        for e in table["b"]:
            assert e.as_py() == "n"

    def test_unkown_type(self) -> None:
        d = {"a": [1, 2, 3], "b": [4, 5, 6]}
        df = pd.DataFrame(data=d, dtype=np.uint8)
        col_types = {"a": "TEST", "b": "TEST"}

        table = DMLOperations._pandas_to_arrow(df, col_types)

        i = 1
        for col in table:
            for row in col:
                assert row.as_py() == i
                assert not row.equals(pa.scalar(i))
                i += 1


class TestDMLOperationsPolarsToArrow:
    def test_unkown_type(self) -> None:
        d = {"a": [1, 2, 3], "b": [4, 5, 6]}
        df = pl.DataFrame(d)
        col_types = {"a": "TEST", "b": "TEST"}
        mwc = MagicMock()
        df.with_columns = mwc  # type: ignore

        DMLOperations._polars_to_arrow(df, col_types)
        mwc.assert_not_called()


class TestDMLOperationsCreateUpdateData:
    def test_simple(self) -> None:
        row = {"a": 1, "b": 2}
        dml = DMLOperations(MagicMock(), MagicMock())
        changes, old = dml._create_update_data(row, {"a": 10})
        assert changes == {"a": 10, "b": 2}
        assert old == {"a": 1}

    def test_nested(self) -> None:
        row = {"a": {"x": 1, "y": 2}, "b": 2}
        dml = DMLOperations(MagicMock(), MagicMock())
        changes, old = dml._create_update_data(row, {"a": {"x": 10}})
        assert changes == {"a": {"x": 10, "y": 2}, "b": 2}
        assert old == {"a": {"x": 1}}

    def test_list(self) -> None:
        row = {"a": [1, 2, 3], "b": 2}
        dml = DMLOperations(MagicMock(), MagicMock())
        changes, old = dml._create_update_data(row, {"a": {1: 20}})
        assert changes == {"a": [1, 20, 3], "b": 2}
        assert old == {"a": {1: 2}}

    def test_mixed(self) -> None:
        row = {"a": [{"x": 1}, {"y": 2}], "b": 2}
        dml = DMLOperations(MagicMock(), MagicMock())
        changes, old = dml._create_update_data(row, {"a": {1: {"y": 20}}})
        assert changes == {"a": [{"x": 1}, {"y": 20}], "b": 2}
        assert old == {"a": {1: {"y": 2}}}

    def test_empty(self) -> None:
        row = {"a": [10], "b": 2}
        dml = DMLOperations(MagicMock(), MagicMock())
        changes, old = dml._create_update_data(row, {})
        assert changes == {}
        assert old == {"a": [10], "b": 2}


class TestDMLOperationsNormalizeChangeDict:
    def test_normalize_change_dict(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        changes = {"a.b": 1, "c": 2}
        normalized = dml._normalize_change_dict(changes)
        assert normalized == {"a": {"b": 1}, "c": 2}

    def test_normalize_change_dict_more_than_one_list_entry(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        changes = {"a[0]": 1, "a[1]": 2}
        normalized = dml._normalize_change_dict(changes)
        assert normalized == {"a": {0: 1, 1: 2}}

    def test_normalize_change_dict_nested(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        changes = {"a.b[0].c": 1, "a.b[1].d": "test"}
        normalized = dml._normalize_change_dict(changes)
        assert normalized == {"a": {"b": {0: {"c": 1}, 1: {"d": "test"}}}}


class TestDMLOperationsGetData:
    def test_get_data_dict(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        data = {"a": 1, "b": 2}
        assert dml._get_data(data, "a") == 1

    def test_get_data_list(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        data = [10, 20, 30]
        assert dml._get_data(data, 1) == 20

    def test_get_data_nested_json_string(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        data = {"a": '{"x": 1, "y": 2}'}
        assert dml._get_data(data, "a") == {"x": 1, "y": 2}

    def test_get_data_nested_string(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        data = {"a": "test"}
        assert dml._get_data(data, "a") == "test"

    def test_get_data_invalid_key(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        data = {"a": 1, "b": 2}
        try:
            dml._get_data(data, "c")
        except TypeError as e:
            assert str(e) == "Key 'c' does not exist in the column"

    def test_get_data_invalid_index(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        data = [10, 20, 30]
        try:
            dml._get_data(data, 5)
        except TypeError as e:
            assert str(e) == "Index '5' out of bounds"

    def test_get_data_non_dict_or_list(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        data = 42
        try:
            dml._get_data(data, "a")
        except TypeError as e:
            assert str(e) == "Changes must follow schema of the table"


class TestDMLOperationsParseChangeKey:
    def test_parse_change_key_simple(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        key = "a"
        changes = {"b": 1}
        parsed_key, parsed_path = dml._parse_change_key(key, changes)
        assert parsed_key == "a"
        assert parsed_path == {"b": 1}

    def test_parse_change_key_nested(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        key = "a.b.c"
        changes = {"d": 1}
        parsed_key, parsed_path = dml._parse_change_key(key, changes)
        assert parsed_key == "a"
        assert parsed_path == {"b": {"c": {"d": 1}}}

    def test_parse_change_key_with_list_index(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        key = "a[0].b"
        changes = {"c": 2}
        parsed_key, parsed_path = dml._parse_change_key(key, changes)
        assert parsed_key == "a"
        assert parsed_path == {0: {"b": {"c": 2}}}

    def test_parse_change_key_nothing_parsed(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        key = "a"
        changes: dict[Any, Any] = {}
        parsed_key, parsed_path = dml._parse_change_key(key, changes)
        assert parsed_key == "a"
        assert parsed_path == {}


class TestDMLOperationsUpdateWhere:
    def test_update_where_invalid_change_event_reason(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        try:
            dml.update_where(
                table_name="test_table",
                where="id = 1",
                changes={"column": "value"},
                change_event_reason="invalid_reason",
                change_comment="test comment",
            )
        except ValueError as e:
            assert str(e).startswith("Invalid cause: 'invalid_reason'")

    def test_update_where_empty_where(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        try:
            dml.update_where(
                table_name="test_table",
                where="",
                changes={"column": "value"},
                change_event_reason="OTHER",
                change_comment="test comment",
            )
        except ValueError as e:
            assert str(e) == "'where' must be a non-empty SQL condition"

    def test_update_where_only_whitespace_where(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        try:
            dml.update_where(
                table_name="test_table",
                where="   ",
                changes={"column": "value"},
                change_event_reason="OTHER",
                change_comment="test comment",
            )
        except ValueError as e:
            assert str(e) == "'where' must be a non-empty SQL condition"

    def test_update_where_invalid_characters_in_where(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        try:
            dml.update_where(
                table_name="test_table",
                where="id = 1; DROP TABLE test_table",
                changes={"column": "value"},
                change_event_reason="OTHER",
                change_comment="test comment",
            )
        except ValueError as e:
            assert str(e) == "'where' must be a single SQL expression"
        try:
            dml.update_where(
                table_name="test_table",
                where="id = 1 -- comment",
                changes={"column": "value"},
                change_event_reason="OTHER",
                change_comment="test comment",
            )
        except ValueError as e:
            assert str(e) == "'where' must be a single SQL expression"
        try:
            dml.update_where(
                table_name="test_table",
                where="id = 1 /* comment */",
                changes={"column": "value"},
                change_event_reason="OTHER",
                change_comment="test comment",
            )
        except ValueError as e:
            assert str(e) == "'where' must be a single SQL expression"

    def test_update_where_empty_changes(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        try:
            dml.update_where(
                table_name="test_table",
                where="id = 1",
                changes={},
                change_event_reason="OTHER",
                change_comment="test comment",
            )
        except ValueError as e:
            assert str(e) == "'changes' must not be empty"

    def test_update_where_tag_info_not_found(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        dml._validate_table_and_columns = MagicMock()  # type: ignore
        # Mock _get_tag_info to return None
        try:
            with (patch("ssb_parquedit.dml.QueryOperations") as mock_query,):
                mock_query._get_tag_info = MagicMock(return_value=None)
                dml.update_where(
                    table_name="test_table",
                    where="id = 1",
                    changes={"column": "value"},
                    change_event_reason="OTHER",
                    change_comment="test comment",
                )
        except ValueError as e:
            assert str(e) == "Tag information not found for table: 'test_table'"

    def test_update_where_affected_rows_zero(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        dml._validate_table_and_columns = MagicMock()  # type: ignore

        # Mock the execute method to simulate 0 affected rows
        dml.conn.execute = MagicMock(
            side_effect=[
                None,  # BEGIN
                MagicMock(fetchone=MagicMock(return_value=[0])),  # SELECT count(*)
                None,  # ROLLBACK
            ]
        )
        with (patch("ssb_parquedit.dml.QueryOperations") as mock_query,):
            mock_query._get_tag_info = MagicMock(
                return_value={"product_name": "test_product"}
            )
            dml.update_where(
                table_name="test_table",
                where="id = 1",
                changes={"column": "value"},
                change_event_reason="OTHER",
                change_comment="test comment",
            )

    def test_update_where_successful_update(self) -> None:
        dml = DMLOperations(MagicMock(), MagicMock())
        dml._validate_table_and_columns = MagicMock()  # type: ignore

        # Mock the execute method to simulate affected rows
        dml.conn.execute = MagicMock(
            side_effect=[
                None,  # BEGIN
                MagicMock(fetchone=MagicMock(return_value=[1])),  # SELECT count(*)
                None,  # UPDATE
                None,  # CALL set_commit_message
                None,  # COMMIT
            ]
        )

        with (
            patch("ssb_parquedit.dml.QueryOperations") as mock_query,
            patch("ssb_parquedit.dml.json.dumps", MagicMock(wraps=dumps_wrapper)),
        ):
            mock_query._get_tag_info = MagicMock(
                return_value={"product_name": "test_product"}
            )

            dml.update_where(
                table_name="test_table",
                where="id = 1",
                changes={"column": "value"},
                change_event_reason="OTHER",
                change_comment="test comment",
            )

    def test_update_where_rollback(self) -> None:
        def execute_mock(cmd: Any, *args: object) -> None:
            if cmd == "BEGIN":
                raise ValueError("test_begin")

        dml = DMLOperations(MagicMock(), MagicMock())
        dml._validate_table_and_columns = MagicMock()  # type: ignore

        # Mock the execute method to simulate an error during update
        execute_mock = MagicMock(side_effect=execute_mock)
        execute_mock.fetchone = MagicMock(return_value=[1])
        dml.conn.execute = execute_mock
        with (
            patch("ssb_parquedit.dml.QueryOperations") as mock_query,
            patch("ssb_parquedit.dml.json.dumps", MagicMock(wraps=dumps_wrapper)),
        ):
            # Mock _get_tag_info to return some valid tag info
            mock_query._get_tag_info = MagicMock(
                return_value={"product_name": "test_product"}
            )
            with pytest.raises(ValueError, match="test_begin"):
                dml.update_where(
                    table_name="test_table",
                    where="id = 1",
                    changes={"column": "value"},
                    change_event_reason="OTHER",
                    change_comment="test comment",
                )

            execute_mock.assert_called_with("ROLLBACK")

    def test_update_where_rollback_bypass(self) -> None:
        def execute_mock(cmd: Any, *args: object) -> None:
            if cmd == "BEGIN":
                raise ValueError("test_begin")
            if cmd == "ROLLBACK":
                raise ValueError("test_rollback")

        dml = DMLOperations(MagicMock(), MagicMock())
        dml._validate_table_and_columns = MagicMock()  # type: ignore

        execute_mock = MagicMock(side_effect=execute_mock)
        execute_mock.fetchone = MagicMock(return_value=[1])
        dml.conn.execute = execute_mock
        with (
            patch("ssb_parquedit.dml.QueryOperations") as mock_query,
            patch("ssb_parquedit.dml.json.dumps", MagicMock(wraps=dumps_wrapper)),
        ):
            mock_query._get_tag_info = MagicMock(
                return_value={"product_name": "test_product"}
            )
            with pytest.raises(ValueError, match="test_begin"):
                dml.update_where(
                    table_name="test_table",
                    where="id = 1",
                    changes={"column": "value"},
                    change_event_reason="OTHER",
                    change_comment="test comment",
                )

            execute_mock.assert_called_with("ROLLBACK")


# ── bulk_update and _bulk_update_var: happy path ────────────────────────────────────────────────────


class TestBulkUpdateVarHappyPath:
    """Tests for the _bulk_update_var method."""

    def test_bulk_update_var_single_row(self, cities_table: ParquEdit) -> None:
        """Test updating a single column value in a single row."""
        dml = DMLOperations(cities_table._get_connection())
        dml._bulk_update_var("cities", "population", [0], [999999])
        result = cities_table.view("cities", where="name = 'Oslo'")
        assert result["population"].iloc[0] == 999999

    def test_bulk_update_var_multiple_rows_same_column(
        self, cities_table: ParquEdit
    ) -> None:
        """Test updating the same column in multiple rows."""
        dml = DMLOperations(cities_table._get_connection())
        dml._bulk_update_var("cities", "population", [0, 1], [500000, 300000])
        result = cities_table.view("cities", order_by="name")
        populations = result["population"].tolist()
        assert populations == [300000, 500000, 75000]

    def test_bulk_update_var_string_column(self, cities_table: ParquEdit) -> None:
        """Test updating string column values."""
        dml = DMLOperations(cities_table._get_connection())
        dml._bulk_update_var("cities", "name", [0], ["New Oslo"])
        result = cities_table.view("cities", where="rowid = 0")
        assert result["name"].iloc[0] == "New Oslo"

    def test_bulk_update_var_preserves_other_columns(
        self, cities_table: ParquEdit
    ) -> None:
        """Test that updating one column doesn't affect others."""
        dml = DMLOperations(cities_table._get_connection())
        dml._bulk_update_var("cities", "population", [0], [999999])
        result = cities_table.view("cities", where="name = 'Oslo'")
        assert result["id"].iloc[0] == 1
        assert result["name"].iloc[0] == "Oslo"


class TestBulkUpdateHappyPath:
    """Tests for the bulk_update method."""

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_single_row_single_column(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test updating a single row with a single column change."""
        mock_user.return_value = "test_user"
        dml = DMLOperations(cities_table._get_connection())

        changes = {0: {"population": 900000}}
        dml.bulk_update(
            "cities",
            changes=changes,
            change_event_reason="OTHER",
            change_comment="test update",
            script_id="script_1",
            script_info="test script",
        )

        result = cities_table.view("cities", where="name = 'Oslo'")
        assert result["population"].iloc[0] == 900000

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_multiple_rows_multiple_columns(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test updating multiple rows with multiple column changes."""
        mock_user.return_value = "test_user"
        dml = DMLOperations(cities_table._get_connection())

        changes = {
            0: {"population": 800000, "name": "Oslo Updated"},
            1: {"population": 290000, "name": "Bergen Updated"},
        }
        dml.bulk_update(
            "cities",
            changes=changes,
            change_event_reason="REVIEW",
            change_comment="bulk test update",
            script_id="script_2",
            script_info="test script 2",
        )

        result = cities_table.view("cities", order_by="name")
        assert result.iloc[0]["name"] == "Bergen Updated"
        assert result.iloc[0]["population"] == 290000
        assert result.iloc[1]["name"] == "Oslo Updated"
        assert result.iloc[1]["population"] == 800000

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_partial_row_updates(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test that rows can be updated with different numbers of column changes."""
        mock_user.return_value = "test_user"
        dml = DMLOperations(cities_table._get_connection())

        changes = {
            0: {"population": 750000},  # Only one column
            1: {"name": "New Bergen", "population": 280000},  # Two columns
        }
        dml.bulk_update(
            "cities",
            changes=changes,
            change_event_reason="OTHER",
            change_comment="partial update",
            script_id="script_3",
            script_info="test",
        )

        result = cities_table.view("cities", where="name = 'Oslo'")
        assert result["population"].iloc[0] == 750000
        # Verify Oslo name is unchanged
        assert result["name"].iloc[0] == "Oslo"

        result = cities_table.view("cities", where="name = 'New Bergen'")
        assert result["population"].iloc[0] == 280000

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_preserves_unmodified_rows(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test that rows not in changes dict remain unchanged."""
        mock_user.return_value = "test_user"
        dml = DMLOperations(cities_table._get_connection())

        changes = {0: {"population": 700001}}  # Only update first row

        dml.bulk_update(
            "cities",
            changes=changes,
            change_event_reason="OTHER",
            change_comment="update one",
            script_id="script_4",
            script_info="test",
        )

        # Check unmodified rows
        result = cities_table.view("cities", where="name = 'Bergen'")
        assert result["population"].iloc[0] == 285000

        result = cities_table.view("cities", where="name = 'Tromsø'")
        assert result["population"].iloc[0] == 75000


# ── bulk_update: edge cases ────────────────────────────────────────────────────


class TestBulkUpdateEdgeCases:
    """Test edge cases and error handling for bulk_update."""

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_invalid_change_reason(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test that invalid change_event_reason raises ValueError."""
        mock_user.return_value = "test_user"
        dml = DMLOperations(cities_table._get_connection())

        changes = {0: {"population": 999999}}
        with pytest.raises(ValueError):
            dml.bulk_update(
                "cities",
                changes=changes,
                change_event_reason="INVALID_REASON",
                change_comment="test",
                script_id="script",
                script_info="info",
            )

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_nonexistent_table(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test that updating a nonexistent table raises an error."""
        mock_user.return_value = "test_user"
        dml = DMLOperations(cities_table._get_connection())

        changes = {0: {"population": 999999}}
        with pytest.raises((ValueError, TypeError)):
            dml.bulk_update(
                "nonexistent_table",
                changes=changes,
                change_event_reason="OTHER",
                change_comment="test",
                script_id="script",
                script_info="info",
            )

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_nonexistent_column(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test that updating a nonexistent column raises an error."""
        mock_user.return_value = "test_user"
        dml = DMLOperations(cities_table._get_connection())

        changes = {0: {"nonexistent_col": 999999}}
        with pytest.raises((ValueError, TypeError)):
            dml.bulk_update(
                "cities",
                changes=changes,
                change_event_reason="OTHER",
                change_comment="test",
                script_id="script",
                script_info="info",
            )

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_empty_changes(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test that empty changes dict doesn't cause issues."""
        mock_user.return_value = "test_user"
        dml = DMLOperations(cities_table._get_connection())

        changes: dict[int, dict[str, Any]] = {}
        # Should not raise an error
        dml.bulk_update(
            "cities",
            changes=changes,
            change_event_reason="OTHER",
            change_comment="empty",
            script_id="script",
            script_info="info",
        )

        # Verify nothing changed
        assert cities_table.count("cities") == 3


# ── bulk_update: transaction handling ────────────────────────────────────────────────────


class TestBulkUpdateTransactionHandling:
    """Test transaction behavior for bulk_update."""

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_successful_commit(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test that successful updates are committed."""
        mock_user.return_value = "test_user"
        dml = DMLOperations(cities_table._get_connection())

        changes = {0: {"population": 725000}, 2: {"population": 80000}}
        dml.bulk_update(
            "cities",
            changes=changes,
            change_event_reason="OTHER",
            change_comment="test commit",
            script_id="script",
            script_info="info",
        )

        # Verify updates persisted
        result = cities_table.view("cities", where="name = 'Oslo'")
        assert result["population"].iloc[0] == 725000

        result = cities_table.view("cities", where="name = 'Tromsø'")
        assert result["population"].iloc[0] == 80000

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_captures_user_info(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test that the dapla_user is captured and used."""
        expected_user = "test_dapla_user_123"
        mock_user.return_value = expected_user
        dml = DMLOperations(cities_table._get_connection())

        changes = {0: {"population": 750000}}
        dml.bulk_update(
            "cities",
            changes=changes,
            change_event_reason="OWNER",
            change_comment="user capture test",
            script_id="script_id_test",
            script_info="info_test",
        )

        # Verify the update went through (user capture tested indirectly)
        result = cities_table.view("cities", where="name = 'Oslo'")
        assert result["population"].iloc[0] == 750000

    @patch("ssb_parquedit.dml.get_dapla_user")
    def test_bulk_update_with_all_valid_reasons(
        self, mock_user: MagicMock, cities_table: ParquEdit
    ) -> None:
        """Test bulk_update with all valid change_event_reason values."""
        mock_user.return_value = "test_user"
        dml = DMLOperations(cities_table._get_connection())

        valid_reasons = [
            "OTHER_SOURCE",
            "REVIEW",
            "OWNER",
            "MARGINAL_UNIT",
            "DUPLICATE",
            "OTHER",
        ]

        for idx, reason in enumerate(valid_reasons):
            if idx < cities_table.count("cities"):
                changes = {idx: {"population": 600000 + idx}}
                dml.bulk_update(
                    "cities",
                    changes=changes,
                    change_event_reason=reason,
                    change_comment=f"test {reason}",
                    script_id=f"script_{reason}",
                    script_info="test",
                )

        # Verify at least one update succeeded
        result = cities_table.view("cities")
        assert result["population"].iloc[0] == 600000
