"""Query operations for DuckDB tables."""

import logging
from typing import Any
from typing import cast

from .utils import SchemaUtils

logger = logging.getLogger(__name__)


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
        limit: int | None = None,
        offset: int = 0,
        columns: list[str] | None = None,
        order_by: str | None = None,
        output_format: str = "pandas",
    ) -> Any:
        """View contents of a table in the DuckLake catalog.

        Args:
            table_name: Name of the table to view.
            limit: Maximum number of rows to return. None returns all rows.
            offset: Number of rows to skip. Defaults to 0. Useful for pagination.
            columns: List of column names to select. None selects all columns (*).
            order_by: ORDER BY clause (without the ORDER BY keyword).
                Example: "created_at DESC" or "name ASC, age DESC"
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
            msg = f"Unknown output_format: {output_format}. Must be 'pandas', 'polars', or 'pyarrow'."
            logger.error(msg)
            raise ValueError(msg)

        SchemaUtils.validate_table_name(table_name)

        # Build SELECT clause
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"

        query = f"SELECT {select_clause} FROM {table_name}"

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit is not None:
            query += f" LIMIT {limit}"
        if offset > 0:
            query += f" OFFSET {offset}"

        result = self.conn.execute(query)

        # Convert to requested format
        if output_format == "pandas":
            return result.df()
        elif output_format == "polars":
            return result.pl()
        elif output_format == "pyarrow":
            return result.arrow()
        else:  # pragma: no cover
            msg = f"Unknown output_format: {output_format}. Must be 'pandas', 'polars', or 'pyarrow'."
            logger.error(msg)
            raise ValueError(msg)

    def count(
        self,
        table_name: str,
    ) -> int:
        """Count rows in a table.

        Args:
            table_name: Name of the table.

        Returns:
            int: Number of rows in the table.

        Example:
            >>> # doctest: +SKIP
            >>> total = query.count("users")
        """
        SchemaUtils.validate_table_name(table_name)

        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.conn.execute(query).df()
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

    def list_tables(self) -> list[str]:
        """List all user-created tables in the DuckLake catalog.

        Uses DuckLake's SHOW TABLES command to retrieve only tables
        created through DuckLake, automatically excluding internal
        metadata and system tables.

        Returns:
            list[str]: A list of user-created table names in the catalog.

        Example:
            >>> # doctest: +SKIP
            >>> tables = query.list_tables()
            >>> print(tables)  # ['products', 'users']
        """
        result = self.conn.execute("SHOW TABLES").df()
        return cast(list[str], result["name"].tolist())
