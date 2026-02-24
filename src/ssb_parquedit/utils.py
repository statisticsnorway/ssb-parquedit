"""Utility functions for schema translation and validation."""

import re
import pandas as pd
from typing import Any


class SQLInjectionError(ValueError):
    """Raised when potentially dangerous SQL patterns are detected."""
    pass


class SQLSanitizer:
    """Sanitization utilities for SQL clauses to prevent injection attacks."""
    
    # Dangerous SQL keywords that shouldn't appear in WHERE or ORDER BY clauses
    DANGEROUS_KEYWORDS = {
        "DROP", "DELETE", "TRUNCATE", "INSERT", "UPDATE", "CREATE",
        "ALTER", "EXEC", "EXECUTE", "UNION", "SELECT", "SCRIPT",
        "JAVASCRIPT", "DECLARE", "CAST", "--", "/*", "*/"
    }
    
    @staticmethod
    def validate_where_clause(where: str | None) -> None:
        """Validate WHERE clause for potential SQL injection patterns.
        
        Args:
            where: WHERE clause condition string.
            
        Raises:
            SQLInjectionError: If dangerous SQL patterns are detected.
            
        Note:
            This provides defense-in-depth but is not a substitute for proper
            parameterization. Always validate user input at the application level.
        """
        if where is None:
            return
            
        where_upper = where.upper().strip()
        
        # Check for dangerous keywords
        for keyword in SQLSanitizer.DANGEROUS_KEYWORDS:
            if keyword in where_upper:
                # Allow CAST in WHERE clauses as it's a normal operation
                if keyword == "CAST":
                    continue
                raise SQLInjectionError(
                    f"Potentially dangerous SQL keyword '{keyword}' detected in WHERE clause"
                )
        
        # Check for comment sequences
        if "--" in where or "/*" in where or "*/" in where:
            raise SQLInjectionError("SQL comment sequences detected in WHERE clause")
    
    @staticmethod
    def validate_order_by_clause(order_by: str | None) -> None:
        """Validate ORDER BY clause for potential SQL injection patterns.
        
        Args:
            order_by: ORDER BY clause string.
            
        Raises:
            SQLInjectionError: If dangerous SQL patterns are detected or invalid format.
            
        Note:
            ORDER BY cannot use parameter binding for column names, so validation
            is more strict. Only alphanumeric column names and ASC/DESC are allowed.
        """
        if order_by is None:
            return
            
        order_by_upper = order_by.upper().strip()
        
        # Check for dangerous keywords and patterns
        for keyword in SQLSanitizer.DANGEROUS_KEYWORDS:
            if keyword in order_by_upper:
                raise SQLInjectionError(
                    f"Potentially dangerous SQL keyword '{keyword}' detected in ORDER BY clause"
                )
        
        # Check for comment sequences
        if "--" in order_by or "/*" in order_by or "*/" in order_by:
            raise SQLInjectionError("SQL comment sequences detected in ORDER BY clause")
        
        # ORDER BY should only contain column names, ASC, DESC, and commas
        # Pattern: column_name [ASC|DESC], column_name [ASC|DESC], ...
        pattern = r"^[\w\s,.()\-\+\*\/]*(?:ASC|DESC)?(?:\s*,\s*[\w\s,.()\-\+\*\/]*(?:ASC|DESC)?)*$"
        if not re.match(pattern, order_by, re.IGNORECASE):
            raise SQLInjectionError(
                f"Invalid ORDER BY clause format: {order_by}. "
                "Only column names, ASC/DESC, and basic operators allowed."
            )
    
    @staticmethod
    def validate_column_list(columns: list[str] | None) -> list[str]:
        """Validate a list of column names.
        
        Args:
            columns: List of column names to validate.
            
        Returns:
            Validated column list.
            
        Raises:
            SQLInjectionError: If any column name is invalid.
        """
        if columns is None:
            return []
        
        for col in columns:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", col):
                raise SQLInjectionError(
                    f"Invalid column name: {col}. "
                    "Column names must start with a letter or underscore "
                    "and contain only alphanumeric characters and underscores."
                )
        return columns


class SchemaUtils:
    """Utilities for schema translation and validation."""
    
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
        """Convert a JSON Schema to a DuckDB CREATE TABLE statement.

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
            'CREATE TABLE users (\\n  id BIGINT NOT NULL,\\n  name VARCHAR\\n);'
        """
        required = set(schema.get("required", []))
        cols = []

        # Stable UUID primary key
        cols.append("_id VARCHAR")    
            
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
            >>> SchemaUtils.validate_table_name("user-table")  # Raises ValueError
        """
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(
                f"Invalid table name: {table_name}. "
                "Table names must start with a letter or underscore, "
                "and contain only letters, numbers, and underscores."
            )

    @staticmethod
    def pandas_to_duckdb(dtype) -> str:
        """Map a pandas dtype to a DuckDB column type."""

        PANDAS_DUCKDB_TYPE_MAP = [
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