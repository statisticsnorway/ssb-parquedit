"""Query operations for DuckDB tables."""

from typing import Any

from .utils import SchemaUtils
from .utils import SQLSanitizer


class QueryOperations:
    """Query operations for reading and analyzing table data.

    This class handles:
    - Table data retrieval with filtering, sorting, and pagination
    - Row counting
    - Table existence checks
    """

    def __init__(self, connection: Any) -> None:
        """Initialize with a DuckDB connection.

        Args:
            connection: DuckDBConnection instance.
        """
        self.conn = connection

    def view(
        self,
        table_name: str,
        limit: int | None,
        offset: int = 0,
        columns: list[str] | None = None,
        filters: dict[str, Any] | list[dict[str, Any]] | None = None,
        order_by: str | None = None,
        output_format: str = "pandas",
    ) -> Any:
        """View contents of a table in the DuckLake catalog.

        Args:
            table_name: Name of the table to view.
            limit: Maximum number of rows to return. None returns all rows. 
            offset: Number of rows to skip. Defaults to 0. Useful for pagination.
            columns: List of column names to select. None selects all columns (*).
            filters: Structured filter conditions. Can be:
                - List of dicts: [{"column": "age", "operator": ">", "value": 25}, ...]
                - Dict with 'and'/'or': {"and": [condition1, condition2]}
                - Single dict condition: {"column": "age", "operator": ">", "value": 25}
                Operators supported: =, !=, <>, <, >, <=, >=, LIKE, IN, NOT IN,
                BETWEEN, IS NULL, IS NOT NULL
            order_by: ORDER BY clause (without the ORDER BY keyword).
                Example: "created_at DESC" or "name ASC, age DESC"
                Only column names and ASC/DESC keywords are allowed.
            output_format: Format for the returned data. Options are:
                - "pandas" (default): Returns pd.DataFrame
                - "polars": Returns pl.DataFrame (requires polars library)
                - "pyarrow": Returns pa.Table (requires pyarrow library)

        Returns:
            Data in the specified format (pandas DataFrame, polars DataFrame, or pyarrow Table).

        Raises:
            ValueError: If output_format is not "pandas", "polars", or "pyarrow".

        Example:
            >>> # doctest: +SKIP
            >>> # Simple view - first 5 rows as pandas DataFrame
            >>> query.view("users", limit=5)

            >>> # Select specific columns
            >>> query.view("users", columns=["id", "name"], limit=10)

            >>> # Filter with structured filters (RECOMMENDED)
            >>> query.view("users", filters={"column": "age", "operator": ">", "value": 25})

            >>> # Multiple conditions with AND
            >>> query.view("users", filters=[
            ...     {"column": "age", "operator": ">", "value": 25},
            ...     {"column": "status", "operator": "=", "value": "active"}
            ... ])

            >>> # Multiple conditions with OR
            >>> query.view("users", filters={
            ...     "or": [
            ...         {"column": "status", "operator": "=", "value": "admin"},
            ...         {"column": "status", "operator": "=", "value": "moderator"}
            ...     ]
            ... })

            >>> # Filter with LIKE operator
            >>> query.view("users", filters={"column": "name", "operator": "LIKE", "value": "%john%"})

            >>> # Filter with IN operator
            >>> query.view("users", filters={"column": "id", "operator": "IN", "value": [1, 2, 3]})

            >>> # Filter with BETWEEN
            >>> query.view("users", filters={"column": "age", "operator": "BETWEEN", "value": [18, 65]})

            >>> # Sort results
            >>> query.view("users", order_by="created_at DESC", limit=10)

            >>> # Pagination
            >>> query.view("users", limit=10, offset=20)  # Page 3

            >>> # Get all rows (no limit)
            >>> query.view("users", limit=None)

            >>> # Return as polars DataFrame
            >>> query.view("users", limit=10, output_format="polars")

            >>> # Return as pyarrow Table
            >>> query.view("users", limit=10, output_format="pyarrow")
        """
        if output_format not in ("pandas", "polars", "pyarrow"):
            raise ValueError(
                f"Unknown output_format: {output_format}. Must be 'pandas', 'polars', or 'pyarrow'."
            )

        SchemaUtils.validate_table_name(table_name)

        # Validate and sanitize SQL clauses to prevent injection
        SQLSanitizer.validate_order_by_clause(order_by)
        if columns:
            SQLSanitizer.validate_column_list(columns)

        # Build parameterized WHERE clause from filters
        where_parameterized, where_params = SQLSanitizer.build_where_from_filters(
            filters
        )

        # Build SELECT clause
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"

        # Build query with parameterized LIMIT and OFFSET
        query = f"SELECT {select_clause} FROM {table_name}"
        params: list[Any] = []

        # Add WHERE clause (now parameterized)
        if where_parameterized:
            query += f" WHERE {where_parameterized}"
            params.extend(where_params)

        # Add ORDER BY clause
        if order_by:
            query += f" ORDER BY {order_by}"

        # Add LIMIT and OFFSET with parameter binding
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        if offset > 0:
            query += " OFFSET ?"
            params.append(offset)

        # Execute with parameterized values
        if params:
            result = self.conn.execute(query, params)
        else:
            result = self.conn.execute(query)

        # Convert to requested format
        if output_format == "pandas":
            return result.df()
        elif output_format == "polars":
            return result.pl()
        elif output_format == "pyarrow":
            return result.arrow()
        else:  # pragma: no cover
            raise ValueError(
                f"Unknown output_format: {output_format}. Must be 'pandas', 'polars', or 'pyarrow'."
            )

    def count(
        self,
        table_name: str,
        filters: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> int:
        """Count rows in a table.

        Args:
            table_name: Name of the table.
            filters: Structured filter conditions. Can be:
                - List of dicts: [{"column": "age", "operator": ">", "value": 25}, ...]
                - Dict with 'and'/'or': {"and": [condition1, condition2]}
                - Single dict condition: {"column": "age", "operator": ">", "value": 25}

        Returns:
            int: Number of rows matching the condition.

        Example:
            >>> # doctest: +SKIP
            >>> # Count all rows
            >>> total = query.count("users") # doctest: +SKIP

            >>> # Count with structured filters (RECOMMENDED)
            >>> active = query.count("users", filters={"column": "status", "operator": "=", "value": "active"}) # doctest: +SKIP

            >>> # Count with complex filters
            >>> recent = query.count("users", filters=[
            ...     {"column": "created_at", "operator": ">", "value": "2024-01-01"},
            ...     {"column": "age", "operator": ">", "value": 18}
            ... ])
        """
        SchemaUtils.validate_table_name(table_name)

        # Build parameterized WHERE clause from filters
        where_parameterized, where_params = SQLSanitizer.build_where_from_filters(
            filters
        )

        query = f"SELECT COUNT(*) as count FROM {table_name}"
        if where_parameterized:
            query += f" WHERE {where_parameterized}"

        result = (
            self.conn.execute(query, where_params).df()
            if where_params
            else self.conn.execute(query).df()
        )
        return int(result["count"].iloc[0])

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the catalog.

        Args:
            table_name: Name of the table to check.

        Returns:
            bool: True if table exists, False otherwise.

        Example:
            >>> # doctest: +SKIP
            >>> if query.table_exists("users"):
            ...     print("Table exists")
        """
        # Validate table name first
        SchemaUtils.validate_table_name(table_name)

        try:
            self.conn.execute(f"SELECT 1 FROM {table_name} WHERE 1=0")
            return True
        except Exception:
            return False
