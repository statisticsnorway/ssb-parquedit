"""Query operations for DuckDB tables."""

import json
import logging
from typing import Any
from typing import cast

import pandas as pd

from .functions import create_config
from .utils import SchemaUtils

logger = logging.getLogger(__name__)


class QueryOperations:
    """Query operations for reading and analyzing table data.

    This class handles:
    - Table data retrieval with filtering, sorting, and pagination
    - Row counting
    - Table existence checks
    """

    def __init__(
        self, connection: Any, db_config: dict[str, str] | None = None
    ) -> None:
        """Initialize with a DuckDB connection.

        Args:
            connection: DuckDBConnection instance.
            db_config: Optional database configuration dict. If None, configuration is auto-detected from the Dapla environment variables.
        """
        self.conn = connection
        self.db_config = db_config or create_config()

    def view(
        self,
        table_name: str,
        where: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        columns: list[str] | None = None,
        order_by: str | None = None,
        output_format: str = "pandas",
    ) -> Any:
        """View contents of a table in the DuckLake catalog.

        Args:
            table_name: Name of the table to view.
            where: Filter condition(s).
            limit: Maximum number of rows to return. None returns all rows.
            offset: Number of rows to skip. Defaults to 0. Useful for pagination.
            columns: List of column names to select. None selects all columns (*).
            order_by: ORDER BY clause (without the ORDER BY keyword). Example: "created_at DESC" or "name ASC, age DESC".
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
            select_clause = "rowid, " + ", ".join(columns)
        else:
            select_clause = "rowid, *"

        query = f"SELECT {select_clause} FROM {table_name}"

        if where:
            query += f" WHERE {where}"

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit is not None:
            query += f" LIMIT {limit}"
        if offset > 0:
            query += f" OFFSET {offset}"

        result = self.conn.execute(query)

        if output_format == "pandas":
            return result.df()
        elif output_format == "polars":
            return result.pl()
        elif output_format == "pyarrow":
            return result.arrow()

    def count(
        self,
        table_name: str,
        where: str | None = None,
    ) -> int:
        """Count rows in a table.

        Args:
            table_name: Name of the table.
            where: Optional SQL WHERE clause to filter results. Defaults to None.

        Returns:
            int: Number of rows in the table.

        Example:
            >>> # doctest: +SKIP
            >>> total = query.count("users")
        """
        SchemaUtils.validate_table_name(table_name)

        query = f"SELECT COUNT(*) as count FROM {table_name}"

        if where:
            query += f" WHERE {where}"

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
        result = self.conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = current_schema()
                AND table_type = 'BASE TABLE'
                AND table_name NOT LIKE 'ducklake_%' --shows up when using local connection(sqllite)
            ORDER BY table_name
            """).df()
        return cast(list[str], result["table_name"].tolist())

    def _get_tag_info(self, table_name: str) -> dict[str, Any] | None:
        result = self.conn.execute(
            "SELECT table_comment FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).df()

        if result.empty or result["TABLE_COMMENT"].iloc[0] is None:
            return None

        return cast(dict[str, Any], json.loads(str(result["TABLE_COMMENT"].iloc[0])))

    def get_edits(self, table_name: str | None = None) -> pd.DataFrame:
        """Retrieve changelog entries from DuckLake snapshots.

        Fetches all snapshots with non-null commit metadata, parses the JSON
        payload in 'commit_extra_info' into separate columns, and optionally
        filters by table name.

        Args:
            table_name: If provided, only returns edits for the given table.
                If None, returns edits for all tables.

        Returns:
            A DataFrame with snapshot data and parsed changelog columns,
            including change_event_reason, changed_by, user_defined_id,
            old_values, new_values, and more.
        """
        df = self.conn.execute(
            "SELECT * FROM snapshots() WHERE commit_extra_info IS NOT NULL;"
        ).df()
        parsed = df["commit_extra_info"].apply(json.loads).apply(pd.Series)
        df = pd.concat([df, parsed], axis=1)

        if table_name:
            filtered = cast(
                pd.DataFrame, df[df["table_name"] == table_name].reset_index(drop=True)
            )
            if filtered.empty:
                logger.warning(
                    f"No edits found for table '{table_name}'. "
                    f"The table may not exist or has no edit history."
                )
            return filtered

        return df
