"""Utility functions for schema translation and validation."""

import logging
import re
from collections.abc import Callable
from typing import Any

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class SchemaUtils:
    """Utilities for schema translation and validation."""

    @staticmethod
    def is_dataframe(value: Any) -> bool:
        """Return whether value is a pandas or polars DataFrame."""
        return isinstance(value, pd.DataFrame | pl.DataFrame)

    @staticmethod
    def translate(prop: dict[str, Any]) -> str:
        """Translate a JSON Schema property to a DuckDB column type.

        Args:
            prop: JSON Schema property definition dictionary.

        Returns:
            str: DuckDB column type specification.

        Example:
            >>> SchemaUtils.translate({"type": "string"})
            'VARCHAR'
            >>> SchemaUtils.translate({"type": "string", "format": "date"})
            'DATE'
        """
        t = prop.get("type")
        if isinstance(t, list):
            # Remove 'null' from union type
            t = next(x for x in t if x != "null")
        if t == "string":
            fmt = prop.get("format")
            if fmt == "date-time":
                return "TIMESTAMP"
            if fmt == "date":
                return "DATE"
            return "VARCHAR"
        if t == "integer":
            return "BIGINT"
        if t == "date-time":
            return "TIMESTAMP"
        if t == "number":
            return "DOUBLE"
        if t == "boolean":
            return "BOOLEAN"
        if t == "array":
            return f"LIST<{SchemaUtils.translate(prop['items'])}>"
        if t == "object":
            props = prop.get("properties")
            if not props:
                return "JSON"
            fields = [f"{k} {SchemaUtils.translate(v)}" for k, v in props.items()]
            return f"STRUCT({', '.join(fields)})"
        return "JSON"

    @staticmethod
    def jsonschema_to_duckdb(schema: dict[str, Any], table_name: str) -> str:
        r"""Convert a JSON Schema to a DuckDB CREATE TABLE statement.

        Args:
            schema: JSON Schema dictionary with 'properties' and optional 'required' fields.
            table_name: Name for the table in the CREATE statement.

        Returns:
            str: DuckDB CREATE TABLE DDL statement.

        Example:
            >>> schema = {
            ...     "properties": {
            ...         "id": {"type": "integer"},
            ...         "name": {"type": "string"}
            ...     },
            ...     "required": ["id"]
            ... }
            >>> SchemaUtils.jsonschema_to_duckdb(schema, "users")
            'CREATE TABLE users (\n  id BIGINT NOT NULL,\n  name VARCHAR\n);'
        """
        required = set(schema.get("required", []))
        cols = []

        for name, prop in schema["properties"].items():

            col = f"{name} {SchemaUtils.translate(prop)}"

            if name in required:
                col += " NOT NULL"
            cols.append(col)
        return f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(cols) + "\n);"

    @staticmethod
    def validate_table_name(table_name: str) -> None:
        """Validate that a table name follows DuckDB naming conventions.

        Args:
            table_name: The table name to validate.

        Raises:
            ValueError: If the table name contains invalid characters.

        Example:
            >>> SchemaUtils.validate_table_name("users")  # OK
            >>> SchemaUtils.validate_table_name("user-table")
            Traceback (most recent call last):
                ...
            ValueError: Invalid table name: user-table. Table names must start with a lowercase letter or underscore, and contain only lowercase letters, numbers, and underscores.
        """
        if not re.match(r"^[a-z_][a-z0-9_]*$", table_name):
            raise ValueError(
                f"Invalid table name: {table_name}. "
                "Table names must start with a lowercase letter or underscore, "
                "and contain only lowercase letters, numbers, and underscores."
            )
        if len(table_name) > 20:
            raise ValueError(
                f"Invalid table name: {table_name}. "
                "Table names must not exceed 20 characters."
            )

    @staticmethod
    def validate_column_names(columns: list[str]) -> None:
        """Validate that column names fit PostgreSQL's identifier length limit.

        PostgreSQL silently truncates identifiers longer than 63 bytes
        (NAMEDATALEN - 1). For a DuckLake catalog backed by Postgres, this can
        make two columns collide or otherwise desync DuckLake's catalog from
        the actual Postgres columns, crashing later reads with
        "Attempted to access index 0 within vector of size 0"
        (see https://github.com/duckdb/ducklake/issues/1089).

        Args:
            columns: Column names to validate.

        Raises:
            ValueError: If any column name exceeds 63 bytes when UTF-8 encoded.

        Example:
            >>> SchemaUtils.validate_column_names(["id", "name"])  # OK
            >>> # doctest: +SKIP
            >>> SchemaUtils.validate_column_names(["a" * 64])
            Traceback (most recent call last):
                ...
            ValueError: Column name(s) exceed PostgreSQL's 63-byte identifier limit...
        """
        too_long = [
            f"{col} ({len(col.encode('utf-8'))} bytes)"
            for col in columns
            if len(col.encode("utf-8")) > 63
        ]
        if too_long:
            raise ValueError(
                "Column name(s) exceed PostgreSQL's 63-byte identifier limit "
                f"and would be silently truncated, corrupting the table: {too_long}. "
                "Shorten these column names before creating the table."
            )

    @staticmethod
    def pandas_to_duckdb(dtype: Any) -> str:
        """Map a pandas dtype to a DuckDB column type."""
        PANDAS_DUCKDB_TYPE_MAP: list[tuple[Callable[[Any], bool], str]] = [
            (lambda d: pd.api.types.is_integer_dtype(d), "BIGINT"),
            (lambda d: pd.api.types.is_float_dtype(d), "DOUBLE"),
            (lambda d: pd.api.types.is_bool_dtype(d), "BOOLEAN"),
            (lambda d: pd.api.types.is_datetime64_any_dtype(d), "TIMESTAMP"),
            (lambda d: pd.api.types.is_string_dtype(d), "VARCHAR"),
            (lambda d: pd.api.types.is_object_dtype(d), "VARCHAR"),
        ]

        for predicate, duck_type in PANDAS_DUCKDB_TYPE_MAP:
            if predicate(dtype):
                return duck_type

        return "VARCHAR"
