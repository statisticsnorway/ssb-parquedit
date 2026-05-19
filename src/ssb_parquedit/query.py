"""Query operations for DuckDB tables."""

import logging
from typing import Any
from typing import cast

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
        result = self.conn.execute("SHOW TABLES").df()
        return cast(list[str], result["name"].tolist())

    def _get_product_name(self, table_name: str) -> str:
        config = self.db_config
        schema = config.get("metadata_schema", config["catalog_name"])

        query = f"""
            SELECT t.value
            FROM __ducklake_metadata_{config["catalog_name"]}.{schema}.ducklake_tag t
            JOIN __ducklake_metadata_{config["catalog_name"]}.{schema}.ducklake_table tb
                ON t.object_id = tb.table_id
            WHERE tb.table_name = '{table_name}'
            AND t.key = 'comment'
            AND t.end_snapshot IS NULL
        """
        result = self.conn.execute(query).df()
        if result.empty:
            return ""
        return str(result["value"].iloc[0])

    def get_edits(self, table_name: str) -> Any:
        """Retrieve historical column-level edits for a specified table within a DuckLake metadata schema.

        This function dynamically determines all columns in the target table, casts them to VARCHAR
        to ensure type consistency during unpivoting, and then compares current and previous values
        across snapshots to identify real data changes.

        Args:
            table_name:
                Name of the target table to inspect for historical edits.

        Returns:
            Any:
                A DuckDB relation or result set containing detected edits, including:
                - `snapshot_id`: Snapshot identifier.
                - `rowid`: Unique identifier within a table
                - `snapshot_time`: Timestamp of the snapshot.
                - `author`: Commit author.
                - `commit_message`: Commit message for the snapshot.
                - `commit_extra_info`: Extra info in addition to commit_message.
                - `change_type`: How a row changed between snapshots.
                - `var`: Column name.
                - `value` / `pre_value`: Current and previous values for that column.
    
        """
        config = self.db_config
        metadata_schema = config.get("metadata_schema", config["catalog_name"])

        # Step 1: Get latest snapshot ID
        max_snapshot_sql = f"""
            SELECT MAX(snapshot_id) AS max_snapshot
            FROM ducklake_snapshots({metadata_schema})
        """
        row = self.conn.execute(max_snapshot_sql).fetchone()
        max_snapshot = row[0] if row is not None else None

        # Step 2: Dynamically determine column names from the table
        # Use DuckDB's DESCRIBE command to get schema info
        describe_df = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()

        cols = [
            c
            for c in describe_df["column_name"].tolist()
            if c not in ("snapshot_id", "snapshot_time", "rowid", "change_type")
        ]

        # Step 3: Generate CAST expressions for consistent UNPIVOT typing
        cast_expr = ", ".join(
            [f"COALESCE(CAST({c} AS VARCHAR),'NULL') AS {c}" for c in cols]
        )

        # Step 4: Construct base changes query

        hist_edit_sql = f"""
        WITH table_changes AS
        (
            SELECT snapshot_id,
                rowid,
                change_type,
                {cast_expr}
            FROM ducklake_table_changes({metadata_schema}, 'main', {table_name} , 0, {max_snapshot})
            WHERE change_type not in ('update_preimage', 'delete')
        ),
        dist_change_types AS
        (
            SELECT DISTINCT snapshot_id, change_type
            FROM table_changes
        ),
        table_edits_unpiv AS
        (
            unpivot table_changes
            ON COLUMNS(* EXCLUDE (snapshot_id, change_type, rowid))
            into name var
                value value
        ),
        column_edits AS
        (
            SELECT snapshot_id, rowid, var, value, LAG(value) OVER (PARTITION BY rowid, var ORDER BY snapshot_id) AS pre_value
            FROM table_edits_unpiv
        )
        SELECT a.*, b.snapshot_time, b.author, b.commit_message, b.commit_extra_info, c.change_type
        FROM column_edits a
        JOIN ducklake_snapshots({metadata_schema}) b
        JOIN dist_change_types c
        ON (b.snapshot_id = a.snapshot_id)
        ON (c.snapshot_id = a.snapshot_id)
        WHERE a.value IS DISTINCT FROM a.pre_value
        AND c.change_type <> 'insert'
        ORDER BY a.snapshot_id, rowid, var
        """

        # Step 6: Execute and return as DuckDB relation
        return self.conn.execute(hist_edit_sql)
