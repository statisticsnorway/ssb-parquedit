"""Tests for SQLSanitizer and SchemaUtils utility classes."""

import pytest

from ssb_parquedit.utils import SQLSanitizer


class TestBuildWhereFromFilters:
    """Test parameterized WHERE clause building from structured filters."""

    def test_none_filters_returns_none(self) -> None:
        where, params = SQLSanitizer.build_where_from_filters(None)
        assert where is None
        assert params == []

    def test_single_condition_equals(self) -> None:
        filters = {"column": "status", "operator": "=", "value": "active"}
        where, params = SQLSanitizer.build_where_from_filters(filters)
        assert where == "status = ?"
        assert params == ["active"]

    def test_single_condition_like(self) -> None:
        filters = {"column": "name", "operator": "LIKE", "value": "%john%"}
        where, params = SQLSanitizer.build_where_from_filters(filters)
        assert where == "name LIKE ?"
        assert params == ["%john%"]

    def test_single_condition_in(self) -> None:
        filters = {"column": "id", "operator": "IN", "value": [1, 2, 3]}
        where, params = SQLSanitizer.build_where_from_filters(filters)
        assert where == "id IN (?, ?, ?)"
        assert params == [1, 2, 3]

    def test_error_unsupported_operator(self) -> None:
        filters = {"column": "age", "operator": "INVALID_OP", "value": 25}
        with pytest.raises(ValueError, match="Unsupported operator"):
            SQLSanitizer.build_where_from_filters(filters)

    def test_error_operator_requires_non_null_value(self) -> None:
        filters = {"column": "age", "operator": ">", "value": None}
        with pytest.raises(ValueError, match="requires a non-null value"):
            SQLSanitizer.build_where_from_filters(filters)

    def test_error_between_requires_two_values(self) -> None:
        filters = {"column": "age", "operator": "BETWEEN", "value": [18]}
        with pytest.raises(ValueError, match="2 values"):
            SQLSanitizer.build_where_from_filters(filters)

    def test_error_filters_must_be_list_or_dict(self) -> None:
        with pytest.raises(TypeError, match="must be None, a list, or a dict"):
            SQLSanitizer.build_where_from_filters("invalid")
