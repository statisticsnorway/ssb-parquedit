"""Utility functions for schema translation and validation."""

import logging
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# Matches either a bare name segment ("address", "city") or a bracketed
# integer index ("[0]", "[12]") within a nested path like "items[1].qty".
_NESTED_PATH_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


class NestedPathUtils:
    """Utilities for reading/writing a single field inside a STRUCT or LIST column.

    A "path" addresses a nested field using dotted/bracket notation, e.g.:
    - "address.city"       -> field "city" inside the STRUCT column "address"
    - "tags[0]"             -> element 0 of the LIST column "tags"
    - "items[1].qty"        -> field "qty" of element 1 of the LIST column "items"

    The base column name is always the first segment of the path.
    """

    @staticmethod
    def parse_path(path: str) -> tuple[str, list[str | int]]:
        """Split a nested path into its base column name and remaining tokens.

        Args:
            path: A column name, optionally followed by nested field/index
                accessors, e.g. "address.city" or "items[1].qty".

        Returns:
            A tuple of (base_column_name, tokens), where tokens is a list of
            str (struct field names) and int (list indices) to apply, in
            order, after the base column. Empty for a plain column path.

        Raises:
            ValueError: If path is empty or contains no valid segments.

        Example:
            >>> NestedPathUtils.parse_path("address.city")
            ('address', ['city'])
            >>> NestedPathUtils.parse_path("items[1].qty")
            ('items', [1, 'qty'])
            >>> NestedPathUtils.parse_path("population")
            ('population', [])
        """
        tokens: list[str | int] = [
            int(idx) if idx else name for name, idx in _NESTED_PATH_RE.findall(path)
        ]
        if not tokens:
            msg = f"Invalid path: '{path}'"
            raise ValueError(msg)
        return tokens[0], tokens[1:]  # type: ignore[return-value]

    @staticmethod
    def to_native(value: Any) -> Any:
        """Recursively convert numpy/pandas values to plain Python types.

        DuckDB's Python parameter binding cannot bind numpy scalar types
        (e.g. numpy.int64) or numpy arrays, which is what DuckDB/pandas use
        to represent STRUCT (dict with numpy scalar values) and LIST
        (numpy.ndarray) column values read back via `.df()`. This makes a
        value safe to pass back into an UPDATE statement.

        Args:
            value: A scalar, dict (STRUCT), or list/numpy.ndarray (LIST)
                value, possibly containing numpy/pandas scalar types.

        Returns:
            The equivalent value using only plain Python types.
        """
        if isinstance(value, dict):
            return {k: NestedPathUtils.to_native(v) for k, v in value.items()}
        if isinstance(value, list | np.ndarray):
            return [NestedPathUtils.to_native(v) for v in value]
        if hasattr(value, "item"):
            return value.item()
        return value

    @staticmethod
    def get_nested(value: Any, tokens: list[str | int]) -> Any:
        """Read the value at a nested path within a STRUCT/LIST value.

        Args:
            value: The container value (dict for STRUCT, list for LIST).
            tokens: Path tokens as returned by `parse_path` (without the base
                column name).

        Returns:
            The value found at the nested path, or None if that part of the
            path doesn't exist yet — either a STRUCT field that is missing
            (or whose parent struct is NULL/None), or a LIST index equal to
            the list's length (i.e. the element doesn't exist yet and would
            be appended by `set_nested`).

        Raises:
            IndexError: If a LIST index is out of range (beyond append range)
                anywhere along the path.
        """
        current = value
        for i, token in enumerate(tokens):
            is_last = i == len(tokens) - 1
            if current is None:
                return None
            if isinstance(token, int):
                if not isinstance(current, list):
                    return None
                if token == len(current) and is_last:
                    return None
                if token >= len(current):
                    msg = (
                        f"List index {token} out of range for list of length "
                        f"{len(current)} (can only append at index "
                        f"{len(current)})."
                    )
                    raise IndexError(msg)
            else:
                if not isinstance(current, dict) or token not in current:
                    return None
            current = current[token]
        return current

    @staticmethod
    def set_nested(value: Any, tokens: list[str | int], new_value: Any) -> Any:
        """Return a copy of `value` with the nested path set to `new_value`.

        If the final token is an integer index equal to the length of the
        target LIST (e.g. index 0 of an empty list), `new_value` is appended
        instead of raising an IndexError, allowing new elements to be
        inserted into empty (or shorter) lists. Likewise, if any STRUCT
        field along the path is missing or NULL/None (including the whole
        `value` itself), it is auto-vivified to an empty dict/list before the
        target field is set, allowing new fields to be inserted into empty
        (or partially NULL) structs.

        Args:
            value: The container value (dict for STRUCT, list for LIST) to
                update. Converted to native Python types and deep-copied
                before mutation; the input is left untouched.
            tokens: Path tokens as returned by `parse_path` (without the base
                column name). Must contain at least one token.
            new_value: The value to assign at the nested path.

        Returns:
            A new container value with the nested path updated.

        Raises:
            IndexError: If a LIST index is out of range (beyond append
                range) anywhere along the path.
        """

        def empty_container_for(token: str | int) -> list[Any] | dict[str, Any]:
            return [] if isinstance(token, int) else {}

        updated = NestedPathUtils.to_native(value)
        if updated is None:
            updated = empty_container_for(tokens[0])
        current = updated
        for token, next_token in zip(tokens[:-1], tokens[1:], strict=True):
            if isinstance(token, int):
                if not isinstance(current, list):
                    msg = f"Expected a LIST to index with [{token}], got {type(current).__name__}."
                    raise TypeError(msg)
                if token > len(current):
                    msg = (
                        f"List index {token} out of range for list of length "
                        f"{len(current)} (can only append at index "
                        f"{len(current)})."
                    )
                    raise IndexError(msg)
                if token == len(current):
                    current.append(empty_container_for(next_token))
                elif current[token] is None:
                    current[token] = empty_container_for(next_token)
            else:
                if not isinstance(current, dict):
                    msg = f"Expected a STRUCT to access field '{token}', got {type(current).__name__}."
                    raise TypeError(msg)
                if current.get(token) is None:
                    current[token] = empty_container_for(next_token)
            current = current[token]
        last = tokens[-1]
        if isinstance(last, int):
            if not isinstance(current, list):
                msg = f"Expected a LIST to index with [{last}], got {type(current).__name__}."
                raise TypeError(msg)
            if last == len(current):
                current.append(new_value)
            elif last > len(current):
                msg = (
                    f"List index {last} out of range for list of length "
                    f"{len(current)} (can only append at index "
                    f"{len(current)})."
                )
                raise IndexError(msg)
            else:
                current[last] = new_value
        else:
            if not isinstance(current, dict):
                msg = f"Expected a STRUCT to access field '{last}', got {type(current).__name__}."
                raise TypeError(msg)
            current[last] = new_value
        return updated


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

        if "rowid" in columns:
            raise ValueError("Column name 'rowid' is reserved and cannot be used.")

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
