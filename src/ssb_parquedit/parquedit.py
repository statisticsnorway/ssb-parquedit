"""ParquEdit - Clean facade for DuckDB table management with DuckLake catalog."""

from typing import Any

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
        """Create a new table. See DDLOperations.create_table for details."""
        if product_name is None or product_name == "":
            raise ValueError(
                "'product_name' must have a value, please provide the valid short-name for your table"
            )

        with self._get_connection() as conn:
            ddl = DDLOperations(conn)
            dml = DMLOperations(conn)

            if (
                isinstance(source, pd.DataFrame)
                or source.__class__.__name__ == "DataFrame"
            ):
                # DuckDB does not support pandas' nullable StringDtype — convert to object dtype first
                source_converted = source.astype(
                    {col: object for col, dtype in source.dtypes.items()
                    if isinstance(dtype, pd.StringDtype)}
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

    # ============ DML Operations ============

    def insert_data(
        self, table_name: str, source: pd.DataFrame | dict[str, Any] | str
    ) -> None:
        """Insert data into a table."""
        with self._get_connection() as conn:
            dml = DMLOperations(conn)
            dml.insert_data(table_name, source)

    # ============ Query Operations ============

    def view(
        self,
        table_name: str,
        limit: int | None = 10,
        offset: int = 0,
        columns: list[str] | None = None,
        filters: dict[str, Any] | list[dict[str, Any]] | None = None,
        order_by: str | None = None,
        output_format: str = "pandas",
    ) -> Any:
        """View table contents. See QueryOperations.view for details."""
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
        """Count table rows. See QueryOperations.count for details."""
        with self._get_connection() as conn:
            query = QueryOperations(conn)
            return query.count(table_name, filters)

    def exists(self, table_name: str) -> bool:
        """Check if table exists. See QueryOperations.table_exists for details."""
        with self._get_connection() as conn:
            query = QueryOperations(conn)
            return query.table_exists(table_name)
