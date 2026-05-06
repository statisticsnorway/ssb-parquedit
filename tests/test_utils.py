"""Tests for schema translation and table name validation."""

import pytest

from ssb_parquedit.utils import SchemaUtils


class TestTranslate:
    """JSON Schema property to DuckDB type translation."""

    def test_string_type(self) -> None:
        assert SchemaUtils.translate({"type": "string"}) == "VARCHAR"

    def test_date_format(self) -> None:
        assert SchemaUtils.translate({"type": "string", "format": "date"}) == "DATE"

    def test_datetime_format(self) -> None:
        assert (
            SchemaUtils.translate({"type": "string", "format": "date-time"})
            == "TIMESTAMP"
        )

    def test_integer_type(self) -> None:
        assert SchemaUtils.translate({"type": "integer"}) == "BIGINT"

    def test_array_type(self) -> None:
        assert (
            SchemaUtils.translate({"type": "array", "items": {"type": "string"}})
            == "LIST<VARCHAR>"
        )

    def test_object_with_properties(self) -> None:
        prop = {"type": "object", "properties": {"x": {"type": "integer"}}}
        result = SchemaUtils.translate(prop)
        assert "STRUCT" in result and "x BIGINT" in result

    def test_nullable_union_drops_null(self) -> None:
        assert SchemaUtils.translate({"type": ["null", "string"]}) == "VARCHAR"

    def test_unknown_type_defaults_to_json(self) -> None:
        assert SchemaUtils.translate({"type": "unknown"}) == "JSON"


class TestJsonschemaToDuckdb:
    """JSON Schema to DuckDB DDL conversion."""

    def test_generates_create_table_with_id_column(self) -> None:
        schema = {
            "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
            "required": ["id"],
        }
        ddl = SchemaUtils.jsonschema_to_duckdb(schema, "users")
        assert "CREATE TABLE users" in ddl
        assert "_id VARCHAR" in ddl
        assert "id BIGINT NOT NULL" in ddl
        assert "name VARCHAR" in ddl


class TestValidateTableName:
    """Table name format validation."""

    def test_valid_names_accepted(self) -> None:
        for name in ("users", "user_data", "_users", "users123"):
            SchemaUtils.validate_table_name(name)  # Should not raise

    def test_uppercase_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name"):
            SchemaUtils.validate_table_name("USERS")

    def test_hyphen_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name"):
            SchemaUtils.validate_table_name("user-data")

    def test_too_long_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name"):
            SchemaUtils.validate_table_name("a" * 21)
