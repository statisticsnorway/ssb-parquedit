"""ParquEdit - Clean facade for DuckDB table management with DuckLake catalog."""

from types import TracebackType
from typing import Any
from typing import Literal

import duckdb
import pandas as pd
from .connection import DuckDBConnection
from .ddl import DDLOperations
from .dml import DMLOperations
from .query import QueryOperations


class ParquEdit:
    """A class for managing DuckDB tables with DuckLake catalog integration.

    This facade provides a unified interface to DDL, DML, and Query operations.

    For detailed documentation of each method, see the respective operations classes:
    - DDL operations: ddl.DDLOperations (create_table, drop_table, alter_table)
    - DML operations: dml.DMLOperations (insert, update, delete, fill_table)
    - Query operations: query.QueryOperations (view, count)

    Example:
        >>> db_config = {
        ...     "dbname": "mydb",
        ...     "dbuser": "user",
        ...     "catalog_name": "my_catalog",
        ...     "data_path": "gs://my-bucket/data",
        ...     "metadata_schema": "public"
        ... }
        >>>
        >>> with ParquEdit(db_config) as editor:
        ...     # Create and populate a table
        ...     editor.create_table("users", df, "User data", fill=True)
        ...
        ...     # Query the table
        ...     result = editor.view("users", limit=10,
        ...         filters={"column": "age", "operator": ">", "value": 25})
    """

    def __init__(
        self, db_config: dict[str, str], conn: duckdb.DuckDBPyConnection | None = None
    ) -> None:
        """Initialize ParquEdit.

        Args:
            db_config: Database configuration dict with keys:
                - dbname: PostgreSQL database name
                - dbuser: PostgreSQL user
                - catalog_name: Name of the DuckLake catalog
                - data_path: Path to data storage (e.g., gs://bucket/path)
                - metadata_schema: PostgreSQL schema for metadata
            conn: Optional existing DuckDB connection to reuse.
        """
        self._connection = DuckDBConnection(db_config, conn)
        self._ddl = DDLOperations(self._connection)
        self._dml = DMLOperations(self._connection)
        self._query = QueryOperations(self._connection)

    def __enter__(self) -> "ParquEdit":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Context manager exit, closes connection if owned."""
        self.close()

    def close(self) -> None:
        """Close the database connection if this instance owns it."""
        self._connection.close()

    # ============ DDL Operations ============
    # Delegate to DDLOperations - see ddl.py for full documentation

    def create_table(
        self,
        table_name: str,
        source: pd.DataFrame | dict[str, Any] | str,
        part_columns: list[str] | None = None,
        fill: bool = False,
    ) -> None:
        """Create a new table. See DDLOperations.create_table for details."""
        self._ddl.create_table(table_name, source, part_columns)
        if fill:
            self._dml.insert_data(table_name, source)

    # ============ DML Operations ============
    # Delegate to DMLOperations - see dml.py for full documentation

    def insert_data(
        self, table_name: str, source: pd.DataFrame | dict[str, Any] | str
    ) -> None:
        """Populate table with data. See DMLOperations.fill_table for details."""
        return self._dml.insert_data(table_name, source)

    # ============ Query Operations ============
    # Delegate to QueryOperations - see query.py for full documentation

    def view(
        self,
        table_name: str,
        limit: int | None = 10,
        offset: int = 0,
        columns: list[str] | None = None,
        filters: dict[str, Any] | list[dict[str, Any]] | None = None,
        order_by: str | None = None,
        output_format: Literal["pandas", "polars", "pyarrow"] = "pandas",
    ) -> Any:
        """View table contents. See QueryOperations.view for details."""
        return self._query.view(
            table_name, limit, offset, columns, filters, order_by, output_format
        )

    def count(
        self,
        table_name: str,
        filters: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> int:
        """Count table rows. See QueryOperations.count for details."""
        return self._query.count(table_name, filters)

    def exists(self, table_name: str) -> bool:
        """Check if table exists. See QueryOperations.table_exists for details."""
        return self._query.table_exists(table_name)
