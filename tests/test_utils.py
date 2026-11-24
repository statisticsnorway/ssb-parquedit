"""Unit tests for SchemaUtils."""

import unittest

import pandas as pd
import polars as pl
import pytest

from ssb_parquedit.utils import ChangeUtils
from ssb_parquedit.utils import PathUtils
from ssb_parquedit.utils import SchemaUtils

# ── is_dataframe ────────────────────────────────────────────────────────────────


class TestIsDataframe:
    """is_dataframe() must recognize both pandas and polars DataFrames."""

    def test_true_for_pandas_dataframe(self) -> None:
        assert SchemaUtils.is_dataframe(pd.DataFrame({"id": [1]})) is True

    def test_true_for_polars_dataframe(self) -> None:
        assert SchemaUtils.is_dataframe(pl.DataFrame({"id": [1]})) is True

    @pytest.mark.parametrize(
        "value", [{"id": [1]}, "gs://bucket/data.parquet", [1, 2], None]
    )
    def test_false_for_non_dataframe_values(self, value: object) -> None:
        assert SchemaUtils.is_dataframe(value) is False


# ── validate_column_names ──────────────────────────────────────────────────────


class TestValidateColumnNames:
    """validate_column_names() enforces PostgreSQL's 63-byte identifier limit."""

    def test_accepts_short_column_names(self) -> None:
        SchemaUtils.validate_column_names(["id", "name", "a" * 63])  # must not raise

    def test_raises_for_column_over_63_bytes(self) -> None:
        with pytest.raises(ValueError, match="63-byte"):
            SchemaUtils.validate_column_names(["a" * 64])

    def test_raises_for_multibyte_column_over_63_bytes(self) -> None:
        """64 chars of Norwegian text can be 67 UTF-8 bytes due to \u00e6/\u00f8/\u00e5."""
        name = "distriktstilskuddforfruktb\u00e6rveksthusgr\u00f8nnsakerinklsalatp\u00e5friland"
        assert len(name) == 64
        assert len(name.encode("utf-8")) == 67
        with pytest.raises(ValueError, match="63-byte"):
            SchemaUtils.validate_column_names([name])

    def test_error_message_reports_byte_length(self) -> None:
        with pytest.raises(ValueError, match=r"64 bytes"):
            SchemaUtils.validate_column_names(["a" * 64])

    def test_error_message_lists_only_offending_columns(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            SchemaUtils.validate_column_names(["short_col", "a" * 64])
        assert "short_col" not in str(exc_info.value)

    def test_rowid_in_columns(self) -> None:
        with pytest.raises(
            ValueError, match=r"Column name 'rowid' is reserved and cannot be used."
        ):
            SchemaUtils.validate_column_names(["rowid"])


class TestTranslate(unittest.TestCase):
    def test_prop_type_is_list_contains_null(self) -> None:
        prop = {"type": ["null", "integer", "null", "number"]}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "BIGINT")

    def test_prop_type_is_string_date_time(self) -> None:
        prop = {"type": "string", "format": "date-time"}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "TIMESTAMP")

    def test_prop_type_is_string_date(self) -> None:
        prop = {"type": "string", "format": "date"}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "DATE")

    def test_prop_type_is_string_char(self) -> None:
        prop = {"type": "string", "format": "char"}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "VARCHAR")

    def test_prop_type_is_integer(self) -> None:
        prop = {"type": "integer"}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "BIGINT")

    def test_prop_type_is_date_time(self) -> None:
        prop = {"type": "date-time"}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "TIMESTAMP")

    def test_prop_type_is_number(self) -> None:
        prop = {"type": "number"}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "DOUBLE")

    def test_prop_type_is_boolean(self) -> None:
        prop = {"type": "boolean"}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "BOOLEAN")

    def test_prop_type_is_array(self) -> None:
        prop = {"type": "array", "items": {"type": "integer"}}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "BIGINT[]")

    def test_prop_type_is_object_with_no_properties(self) -> None:
        prop = {"type": "object"}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "JSON")

    def test_prop_type_is_object_with_properties(self) -> None:
        prop = {"type": "object", "properties": {"a": {"type": "integer"}}}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "STRUCT(a BIGINT)")

    def test_prop_type_is_unknown(self) -> None:
        prop = {"type": "unknown"}
        out = SchemaUtils.translate(prop)
        self.assertEqual(out, "JSON")


class TestJsonschemaToDuckDb(unittest.TestCase):
    def test_name_in_required(self) -> None:
        schema = {
            "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
            "required": ["name"],
        }

        out = SchemaUtils.jsonschema_to_duckdb(schema, "t1")
        self.assertIn("name VARCHAR NOT NULL", out)


class TestValidateTableName(unittest.TestCase):
    def test_invalid_characters_in_table_name(self) -> None:
        table_name = "INVALID"
        with pytest.raises(
            ValueError,
            match=f"Invalid table name: {table_name}. "
            "Table names must start with a lowercase letter or underscore, "
            "and contain only lowercase letters, numbers, and underscores.",
        ):
            SchemaUtils.validate_table_name(table_name)

    def test_table_name_to_long(self) -> None:
        table_name = "a" * 21
        with pytest.raises(
            ValueError,
            match=f"Invalid table name: {table_name}. "
            "Table names must not exceed 20 characters.",
        ):
            SchemaUtils.validate_table_name(table_name)


class TestPandasToDuckDB(unittest.TestCase):
    def test_dtype_is_integer(self) -> None:
        out = SchemaUtils.pandas_to_duckdb(pd.Int32Dtype)
        self.assertEqual(out, "BIGINT")

    def test_dtype_is_float(self) -> None:
        out = SchemaUtils.pandas_to_duckdb(pd.Float32Dtype)
        self.assertEqual(out, "DOUBLE")

    def test_dtype_is_boolean(self) -> None:
        out = SchemaUtils.pandas_to_duckdb(pd.BooleanDtype())
        self.assertEqual(out, "BOOLEAN")

    def test_dtype_is_datetime(self) -> None:
        out = SchemaUtils.pandas_to_duckdb(pd.DatetimeTZDtype(tz="UTC"))
        self.assertEqual(out, "TIMESTAMP")

    def test_dtype_is_string(self) -> None:
        out = SchemaUtils.pandas_to_duckdb(pd.StringDtype())
        self.assertEqual(out, "VARCHAR")

    def test_dtype_is_object(self) -> None:
        out = SchemaUtils.pandas_to_duckdb({})
        self.assertEqual(out, "VARCHAR")

    def test_dtype_is_unkown(self) -> None:
        out = SchemaUtils.pandas_to_duckdb(None)
        self.assertEqual(out, "VARCHAR")


class TestPathUtilsParsePath(unittest.TestCase):
    def test_simple_path(self) -> None:
        path = "population"
        key, lst = PathUtils.parse_path(path)
        self.assertEqual(key, path)
        self.assertEqual(len(lst), 0)

    def test_simple_list(self) -> None:
        path = "a[0]"
        key, lst = PathUtils.parse_path(path)
        self.assertEqual(key, "a")
        self.assertEqual(len(lst), 1)
        self.assertEqual(lst[0], 0)

    def test_dot_notation(self) -> None:
        path = "a.b.c"
        key, lst = PathUtils.parse_path(path)
        self.assertEqual(key, "a")
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], "b")
        self.assertEqual(lst[1], "c")

    def test_dot_list(self) -> None:
        path = "a.b[0].c"
        key, lst = PathUtils.parse_path(path)
        self.assertEqual(key, "a")
        self.assertEqual(len(lst), 3)
        self.assertEqual(lst[0], "b")
        self.assertEqual(lst[1], 0)
        self.assertEqual(lst[2], "c")

    def test_invalid_path(self) -> None:
        path = ""
        with pytest.raises(ValueError, match="Invalid path: ''"):
            PathUtils.parse_path(path)

class TestChangeUtilsNormalizeChangeDict:
    def test_normalize_change_dict(self) -> None:
        changes = {"a.b": 1, "c": 2}
        normalized = ChangeUtils.normalize_change_dict(changes)
        assert normalized == {"a": {"b": 1}, "c": 2}

    def test_normalize_change_dict_more_than_one_list_entry(self) -> None:
        changes = {"a[0]": 1, "a[1]": 2}
        normalized = ChangeUtils.normalize_change_dict(changes)
        assert normalized == {"a": {0: 1, 1: 2}}

    def test_normalize_change_dict_nested(self) -> None:
        changes = {"a.b[0].c": 1, "a.b[1].d": "test"}
        normalized = ChangeUtils.normalize_change_dict(changes)
        assert normalized == {"a": {"b": {0: {"c": 1}, 1: {"d": "test"}}}}


class TestChangeUtilsGetData:
    def test_get_data_dict(self) -> None:
        data = {"a": 1, "b": 2}
        assert ChangeUtils.get_data(data, "a") == 1

    def test_get_data_list(self) -> None:
        data = [10, 20, 30]
        assert ChangeUtils.get_data(data, 1) == 20

    def test_get_data_nested_json_string(self) -> None:
        data = {"a": '{"x": 1, "y": 2}'}
        assert ChangeUtils.get_data(data, "a") == {"x": 1, "y": 2}

    def test_get_data_nested_string(self) -> None:
        data = {"a": "test"}
        assert ChangeUtils.get_data(data, "a") == "test"

    def test_get_data_invalid_key(self) -> None:
        data = {"a": 1, "b": 2}
        try:
            ChangeUtils.get_data(data, "c")
        except TypeError as e:
            assert str(e) == "Key 'c' does not exist in the column"

    def test_get_data_invalid_index(self) -> None:
        data = [10, 20, 30]
        try:
            ChangeUtils.get_data(data, 5)
        except TypeError as e:
            assert str(e) == "Index '5' out of bounds"

    def test_get_data_non_dict_or_list(self) -> None:
        data = 42
        try:
            ChangeUtils.get_data(data, "a")
        except TypeError as e:
            assert str(e) == "Changes must follow schema of the table"


class TestChangeUtilsParseChangeKey:
    def test_parse_change_key_simple(self) -> None:
        key = "a"
        changes = {"b": 1}
        parsed_key, parsed_path = ChangeUtils.parse_change_key(key, changes)
        assert parsed_key == "a"
        assert parsed_path == {"b": 1}

    def test_parse_change_key_nested(self) -> None:
        key = "a.b.c"
        changes = {"d": 1}
        parsed_key, parsed_path = ChangeUtils.parse_change_key(key, changes)
        assert parsed_key == "a"
        assert parsed_path == {"b": {"c": {"d": 1}}}

    def test_parse_change_key_with_list_index(self) -> None:
        key = "a[0].b"
        changes = {"c": 2}
        parsed_key, parsed_path = ChangeUtils.parse_change_key(key, changes)
        assert parsed_key == "a"
        assert parsed_path == {0: {"b": {"c": 2}}}

    def test_parse_change_key_nothing_parsed(self) -> None:
        key = "a"
        changes: dict[Any, Any] = {}
        parsed_key, parsed_path = ChangeUtils.parse_change_key(key, changes)
        assert parsed_key == "a"
        assert parsed_path == {}