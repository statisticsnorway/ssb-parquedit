"""Tests for SQLSanitizer and SchemaUtils utility classes."""
# NOTE: Private helper methods (_build_comparison_condition, etc.) are covered
# implicitly via the public build_where_from_filters API and not tested separately.

from typing import Any
from typing import cast

import pytest

from ssb_parquedit.utils import SchemaUtils
from ssb_parquedit.utils import SQLInjectionError
from ssb_parquedit.utils import SQLSanitizer

# ================== SQLSanitizer Tests ==================


class TestValidateOrderByClause:
    """Test SQL injection prevention for ORDER BY clauses."""

    def test_none_order_by_is_valid(self) -> None:
        """Test that None is a valid ORDER BY clause."""
        SQLSanitizer.validate_order_by_clause(None)  # Should not raise

    def test_simple_column_asc_is_valid(self) -> None:
        """Test basic ORDER BY with single column ASC."""
        SQLSanitizer.validate_order_by_clause("name ASC")

    def test_dangerous_keyword_drop_raises_error(self) -> None:
        """Test that DROP keyword in ORDER BY raises error."""
        with pytest.raises(SQLInjectionError, match="DROP"):
            SQLSanitizer.validate_order_by_clause("DROP TABLE users")


class TestValidateColumnList:
    """Test column name validation."""

    def test_none_column_list_returns_empty_list(self) -> None:
        """Test that None columns returns empty list."""
        result = SQLSanitizer.validate_column_list(None)
        assert result == []

    def test_invalid_column_name_with_dash(self) -> None:
        """Test that column names with dashes are rejected."""
        with pytest.raises(SQLInjectionError, match="Invalid column name"):
            SQLSanitizer.validate_column_list(["user-name"])


class TestBuildWhereFromFilters:
    """Test parameterized WHERE clause building from structured filters."""

    def test_none_filters_returns_none(self) -> None:
        """Test that None filters returns (None, [])."""
        where, params = SQLSanitizer.build_where_from_filters(None)
        assert where is None
        assert params == []

    def test_single_condition_equals(self) -> None:
        """Test single condition with = operator."""
        filters = {"column": "status", "operator": "=", "value": "active"}
        where, params = SQLSanitizer.build_where_from_filters(filters)
        assert where == "status = ?"
        assert params == ["active"]

    def test_multiple_conditions_or_logic(self) -> None:
        """Test OR logic in dict form."""
        filters = {
            "or": [
                {"column": "status", "operator": "=", "value": "admin"},
                {"column": "status", "operator": "=", "value": "moderator"},
            ]
        }
        where, params = SQLSanitizer.build_where_from_filters(filters)
        assert where == "status = ? OR status = ?"
        assert params == ["admin", "moderator"]

    def test_error_invalid_column_name(self) -> None:
        """Test that invalid column names raise SQLInjectionError."""
        filters = {"column": "user; DROP TABLE users", "operator": "=", "value": "x"}
        with pytest.raises(SQLInjectionError, match="Invalid column name"):
            SQLSanitizer.build_where_from_filters(filters)

    def test_error_unsupported_operator(self) -> None:
        """Test that unsupported operators raise ValueError."""
        filters = {"column": "age", "operator": "INVALID_OP", "value": 25}
        with pytest.raises(ValueError, match="Unsupported operator"):
            SQLSanitizer.build_where_from_filters(filters)

    def test_error_filters_must_be_list_or_dict(self) -> None:
        """Test that filters must be list or dict."""
        with pytest.raises(TypeError, match="must be None, a list, or a dict"):
            SQLSanitizer.build_where_from_filters("invalid")

    def test_error_condition_must_be_dict(self) -> None:
        """Test that each condition must be a dict."""
        filters = ["not a dict"]
        with pytest.raises(TypeError, match="must be a dict"):
            SQLSanitizer.build_where_from_filters(cast(Any, filters))


class _REMOVED_PLACEHOLDER:

    def test_validate_column_name_invalid_starts_with_digit(self) -> None:
        """Test column names starting with digit are invalid."""
        with pytest.raises(SQLInjectionError, match="Invalid column name"):
            SQLSanitizer._validate_column_name("1user")

    def test_validate_column_name_invalid_with_special_chars(self) -> None:
        """Test column names with special characters are invalid."""
        with pytest.raises(SQLInjectionError, match="Invalid column name"):
            SQLSanitizer._validate_column_name("user-name")

    def test_validate_column_name_none_is_invalid(self) -> None:
        """Test None column name is invalid."""
        with pytest.raises(SQLInjectionError, match="Invalid column name"):
            SQLSanitizer._validate_column_name(None)

    def test_validate_column_name_empty_string_is_invalid(self) -> None:
        """Test empty string is invalid."""
        with pytest.raises(SQLInjectionError, match="Invalid column name"):
            SQLSanitizer._validate_column_name("")

    # Tests for _build_comparison_condition
    def test_build_comparison_condition_equals(self) -> None:
        """Test building = condition."""
        params: list[Any] = []
        result = SQLSanitizer._build_comparison_condition("age", "=", 25, params)
        assert result == "age = ?"
        assert params == [25]

    def test_build_comparison_condition_greater_than(self) -> None:
        """Test building > condition."""
        params: list[Any] = []
        result = SQLSanitizer._build_comparison_condition("price", ">", 100.50, params)
        assert result == "price > ?"
        assert params == [100.50]

    def test_build_comparison_condition_null_value_raises_error(self) -> None:
        """Test that null values raise error for comparison operators."""
        params: list[Any] = []
        with pytest.raises(ValueError, match="requires a non-null value"):
            SQLSanitizer._build_comparison_condition("age", "=", None, params)

    # Tests for _build_like_condition
    def test_build_like_condition_valid(self) -> None:
        """Test building valid LIKE condition."""
        params: list[Any] = []
        result = SQLSanitizer._build_like_condition("name", "%john%", params)
        assert result == "name LIKE ?"
        assert params == ["%john%"]

    def test_build_like_condition_non_string_raises_error(self) -> None:
        """Test that non-string value raises error for LIKE."""
        params: list[Any] = []
        with pytest.raises(ValueError, match="requires a string value"):
            SQLSanitizer._build_like_condition("name", 123, params)

    # Tests for _build_in_condition
    def test_build_in_condition_valid_list(self) -> None:
        """Test building valid IN condition with list."""
        params: list[Any] = []
        result = SQLSanitizer._build_in_condition("id", "IN", [1, 2, 3], params)
        assert result == "id IN (?, ?, ?)"
        assert params == [1, 2, 3]

    def test_build_in_condition_valid_tuple(self) -> None:
        """Test building valid IN condition with tuple."""
        params: list[Any] = []
        result = SQLSanitizer._build_in_condition(
            "status", "NOT IN", ("a", "b"), params
        )
        assert result == "status NOT IN (?, ?)"
        assert params == ["a", "b"]

    def test_build_in_condition_non_list_raises_error(self) -> None:
        """Test that non-list value raises error for IN."""
        params: list[Any] = []
        with pytest.raises(ValueError, match="requires a list/tuple value"):
            SQLSanitizer._build_in_condition("id", "IN", 123, params)

    def test_build_in_condition_empty_list_raises_error(self) -> None:
        """Test that empty list raises error for IN."""
        params: list[Any] = []
        with pytest.raises(ValueError, match="requires a non-empty list"):
            SQLSanitizer._build_in_condition("id", "IN", [], params)

    # Tests for _build_between_condition
    def test_build_between_condition_valid(self) -> None:
        """Test building valid BETWEEN condition."""
        params: list[Any] = []
        result = SQLSanitizer._build_between_condition("age", [18, 65], params)
        assert result == "age BETWEEN ? AND ?"
        assert params == [18, 65]

    def test_build_between_condition_tuple(self) -> None:
        """Test BETWEEN with tuple."""
        params: list[Any] = []
        result = SQLSanitizer._build_between_condition("price", (10.5, 99.99), params)
        assert result == "price BETWEEN ? AND ?"
        assert params == [10.5, 99.99]

    def test_build_between_condition_wrong_count_raises_error(self) -> None:
        """Test that BETWEEN with wrong value count raises error."""
        params: list[Any] = []
        with pytest.raises(ValueError, match="2 values"):
            SQLSanitizer._build_between_condition("age", [18], params)

    def test_build_between_condition_non_list_raises_error(self) -> None:
        """Test that BETWEEN with non-list raises error."""
        params: list[Any] = []
        with pytest.raises(ValueError, match="2 values"):
            SQLSanitizer._build_between_condition("age", "invalid", params)

    # Tests for _process_single_condition
    def test_process_single_condition_comparison(self) -> None:
        """Test processing comparison condition."""
        params: list[Any] = []
        condition = {"column": "age", "operator": ">", "value": 30}
        result = SQLSanitizer._process_single_condition(condition, params)
        assert result == "age > ?"
        assert params == [30]

    def test_process_single_condition_like(self) -> None:
        """Test processing LIKE condition."""
        params: list[Any] = []
        condition = {"column": "email", "operator": "LIKE", "value": "%@example.com"}
        result = SQLSanitizer._process_single_condition(condition, params)
        assert result == "email LIKE ?"
        assert params == ["%@example.com"]

    def test_process_single_condition_in(self) -> None:
        """Test processing IN condition."""
        params: list[Any] = []
        condition = {"column": "role", "operator": "IN", "value": ["admin", "user"]}
        result = SQLSanitizer._process_single_condition(condition, params)
        assert result == "role IN (?, ?)"
        assert params == ["admin", "user"]

    def test_process_single_condition_is_null(self) -> None:
        """Test processing IS NULL condition."""
        params: list[Any] = []
        condition = {"column": "deleted_at", "operator": "IS NULL"}
        result = SQLSanitizer._process_single_condition(condition, params)
        assert result == "deleted_at IS NULL"
        assert params == []

    def test_process_single_condition_is_not_null(self) -> None:
        """Test processing IS NOT NULL condition."""
        params: list[Any] = []
        condition = {"column": "updated_at", "operator": "IS NOT NULL"}
        result = SQLSanitizer._process_single_condition(condition, params)
        assert result == "updated_at IS NOT NULL"
        assert params == []

    def test_process_single_condition_not_dict_raises_error(self) -> None:
        """Test that non-dict condition raises error."""
        params: list[Any] = []
        with pytest.raises(TypeError, match="must be a dict"):
            SQLSanitizer._process_single_condition(cast(Any, "invalid"), params)

    def test_process_single_condition_invalid_column_raises_error(self) -> None:
        """Test that invalid column name raises error."""
        params: list[Any] = []
        condition = {"column": "col;DROP", "operator": "=", "value": "x"}
        with pytest.raises(SQLInjectionError, match="Invalid column name"):
            SQLSanitizer._process_single_condition(condition, params)

    # Tests for _extract_filters
    def test_extract_filters_from_list(self) -> None:
        """Test extracting filters from list."""
        filters = [
            {"column": "age", "operator": ">", "value": 25},
            {"column": "status", "operator": "=", "value": "active"},
        ]
        conditions, logic = SQLSanitizer._extract_filters(filters)
        assert conditions == filters
        assert logic == "AND"

    def test_extract_filters_from_dict_with_and(self) -> None:
        """Test extracting filters from dict with 'and' key."""
        condition_list = [
            {"column": "age", "operator": ">", "value": 25},
        ]
        filters = {"and": condition_list}
        conditions, logic = SQLSanitizer._extract_filters(filters)
        assert conditions == condition_list
        assert logic == "AND"

    def test_extract_filters_from_dict_with_or(self) -> None:
        """Test extracting filters from dict with 'or' key."""
        condition_list = [
            {"column": "status", "operator": "=", "value": "admin"},
        ]
        filters = {"or": condition_list}
        conditions, logic = SQLSanitizer._extract_filters(filters)
        assert conditions == condition_list
        assert logic == "OR"

    def test_extract_filters_single_condition_dict(self) -> None:
        """Test extracting single condition dict."""
        filters = {"column": "id", "operator": "=", "value": 1}
        conditions, logic = SQLSanitizer._extract_filters(filters)
        assert conditions == [filters]
        assert logic == "AND"

    def test_extract_filters_invalid_dict_raises_error(self) -> None:
        """Test that dict without and/or/column raises error."""
        filters = {"invalid_key": "value"}
        with pytest.raises(ValueError, match="must have 'and'/'or' key"):
            SQLSanitizer._extract_filters(filters)

    def test_extract_filters_invalid_type_raises_error(self) -> None:
        """Test that non-dict/list raises error."""
        with pytest.raises(TypeError, match="must be None, a list, or a dict"):
            SQLSanitizer._extract_filters("invalid")


# ================== SchemaUtils Tests ==================


class TestTranslate:
    """Test JSON Schema property to DuckDB type translation."""

    def test_string_type_basic(self) -> None:
        """Test basic string type translation."""
        assert SchemaUtils.translate({"type": "string"}) == "VARCHAR"

    def test_string_type_with_date_format(self) -> None:
        """Test string type with date format."""
        assert SchemaUtils.translate({"type": "string", "format": "date"}) == "DATE"



class TestJsonschemaToducKdb:
    """Test JSON Schema to DuckDB DDL conversion."""

    def test_simple_schema_conversion(self) -> None:
        """Test conversion of simple schema."""
        schema = {
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["id"],
        }
        ddl = SchemaUtils.jsonschema_to_duckdb(schema, "users")
        assert "CREATE TABLE users" in ddl
        assert "id BIGINT NOT NULL" in ddl
        assert "name VARCHAR" in ddl
        assert "_id VARCHAR" in ddl
        assert ddl.strip().endswith(");")



class TestValidateTableName:
    """Test table name validation."""

    def test_valid_table_name_lowercase(self) -> None:
        """Test valid lowercase table name."""
        SchemaUtils.validate_table_name("users")



class TestPandasToDuckdb:
    """Test pandas dtype to DuckDB type mapping."""

    # Note: These tests require real pandas functionality
    # The conftest.py stubs pandas with only a DataFrame class
    # So we skip dtype testing and instead document the function exists

    def test_pandas_to_duckdb_function_exists(self) -> None:
        """Test that pandas_to_duckdb function is available."""
        assert hasattr(SchemaUtils, "pandas_to_duckdb")
        assert callable(SchemaUtils.pandas_to_duckdb)
