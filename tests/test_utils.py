"""Unit tests for SchemaUtils."""

import unittest

import pandas as pd
import polars as pl
import pytest

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
        self.assertEqual(out, "LIST<BIGINT>")

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
