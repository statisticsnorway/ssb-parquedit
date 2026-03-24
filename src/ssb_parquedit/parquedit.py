"""ParquEdit - Clean facade for DuckDB table management with DuckLake catalog."""

from typing import Any
from typing import cast

import pandas as pd

from .connection import DuckDBConnection
from .ddl import DDLOperations
from .dml import DMLOperations
from .functions import create_config
from .query import QueryOperations


class ParquEdit:
    """A class for managing DuckDB tables with DuckLake catalog integration.

    This facade provides a unified interface to DDL, DML, and Query operations.
    Each method opens and closes its own connection automatically.

    """

    def __init__(self, config: dict[str, str] | None = None) -> None:
        """Initialize ParquEdit."""
        self._db_config = config if config is not None else create_config()

    def _get_connection(self) -> DuckDBConnection:
        """Create a new connection."""
        return DuckDBConnection(self._db_config)

    # ============ DDL Operations ============

    def create_table(
        self,
        table_name: str,
        source: pd.DataFrame | dict[str, Any] | str,
        product_name: str | None = None,
        part_columns: list[str] | None = None,
        fill: bool = False,
    ) -> None:
        """Create a new table in the DuckLake catalog.

        Args:
            table_name: Name of the table to create. Must be lowercase, start
                with a letter or underscore, and contain only lowercase letters,
                numbers, and underscores. Maximum 20 characters.
            source: Source for the table schema. Can be:
                - pd.DataFrame: Creates table structure from the DataFrame schema.
                - dict: JSON Schema specification defining the table structure.
                - str: GCS path (gs://) to a Parquet file to infer schema from.
            product_name: Label identifying the product this table belongs to.
                Stored as a comment on the table. Must not be None or empty.
            part_columns: Optional list of column names to partition the table by.
            fill: If True, inserts data from source into the table immediately
                after creation. Defaults to False.

        Raises:
            ValueError: If product_name is None or empty.
        """
        if product_name is None or product_name == "":
            raise ValueError(
                "'product_name' must have a value, please provide the valid product-name for your table"
            )

        with self._get_connection() as conn:
            ddl = DDLOperations(conn)
            dml = DMLOperations(conn)

            if (
                isinstance(source, pd.DataFrame)
                or source.__class__.__name__ == "DataFrame"
            ):
                # DuckDB does not support pandas' nullable StringDtype — convert to object dtype first
                df = cast(pd.DataFrame, source)
                source_converted = df.astype(
                    {
                        col: object
                        for col, dtype in df.dtypes.items()
                        if isinstance(dtype, pd.StringDtype)
                    }
                )
                conn._conn.register("data", source_converted)
                conn._conn.execute(
                    f"CREATE TABLE {table_name} AS "
                    f"SELECT CAST(NULL AS VARCHAR) AS _id, * FROM data WHERE 1=2"
                )

            elif isinstance(source, dict):
                ddl.create_table(table_name, source, part_columns)
            elif isinstance(source, str):
                ddl.create_table(table_name, source)
            else:
                ddl.create_table(table_name, source, part_columns)

            if part_columns and len(part_columns) > 0:
                columns_str = ",".join(part_columns)
                conn._conn.execute(
                    f"ALTER TABLE {table_name} SET PARTITIONED BY ({columns_str});"
                )

            if fill:
                dml.insert_data(table_name, source)

            conn._conn.execute(f"COMMENT ON TABLE {table_name} IS '{product_name}';")

    def drop_table(self, table_name: str) -> None:
        """Drop a table from the DuckLake catalog.

        Table deletion is only allowed in the TEST environment to prevent
        accidental data loss in production. In PROD or other environments,
        this method will raise a PermissionError.

        Args:
            table_name: Name of the table to drop.

        Raises:
            PermissionError: If DAPLA_ENVIRONMENT is not "test".
            ValueError: If table_name is invalid.

        Example:
            >>> # doctest: +SKIP
            >>> con = ParquEdit()
            >>> con.drop_table("temporary_table")  # Only works in TEST environment
        """
        with self._get_connection() as conn:
            ddl = DDLOperations(conn)
            ddl.drop_table(table_name)

    # ============ DML Operations ============

    def insert_data(
        self, table_name: str, source: pd.DataFrame | dict[str, Any] | str
    ) -> None:
        """Insert data into a table.

        Args:
            table_name: The name of the table to insert data into.
            source: The data to insert.
                Can be a pandas DataFrame, a dictionary mapping column names
                to values, or a string file path to a data file.
        """
        with self._get_connection() as conn:
            dml = DMLOperations(conn)
            dml.insert_data(table_name, source)

    # ============ Query Operations ============

    def view(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int = 0,
        columns: list[str] | None = None,
        filters: dict[str, Any] | list[dict[str, Any]] | None = None,
        order_by: str | None = None,
        output_format: str = "pandas",
    ) -> Any:
        """View the contents of a table.

        Args:
            table_name: The name of the table to query.
            limit: Maximum number of rows to return. Defaults to None.
            offset: Number of rows to skip before returning results.
                Defaults to 0.
            columns: List of column names to include. Defaults
                to None, which returns all columns.
            filters: Filter
                conditions to apply. Can be a single filter dict or a list of
                filter dicts. Defaults to None.
            order_by: Column name to sort results by. Defaults
                to None.
            output_format: Format of the returned data. Defaults to
                "pandas".

        Returns:
            Any: Query results in the specified output format.
        """
        with self._get_connection() as conn:
            query = QueryOperations(conn)
            return query.view(
                table_name,
                limit=limit,
                offset=offset,
                columns=columns,
                filters=filters,
                order_by=order_by,
                output_format=output_format,
            )

    def count(
        self,
        table_name: str,
        filters: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> int:
        """Count the number of rows in a table.

        Args:
            table_name: The name of the table to count rows in.
            filters: Filter
                conditions to apply before counting. Can be a single filter
                dict or a list of filter dicts. Defaults to None.

        Returns:
            int: The number of rows matching the given filters.
        """
        with self._get_connection() as conn:
            query = QueryOperations(conn)
            return query.count(table_name, filters)

    def exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: The name of the table to check for existence.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        with self._get_connection() as conn:
            query = QueryOperations(conn)
            return query.table_exists(table_name)

    def list_tables(self) -> list[str]:
        """List all tables in the current catalog.

        Returns:
            list[str]: A list of table names in the catalog, sorted alphabetically.
        """
        with self._get_connection() as conn:
            query = QueryOperations(conn)
            return query.list_tables()
