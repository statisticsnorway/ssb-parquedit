"""ParquEdit - Clean facade for DuckDB table management with DuckLake catalog."""

from types import TracebackType
from typing import Any
import duckdb
import pandas as pd

from connection import DuckDBConnection
from ddl import DDLOperations
from dml import DMLOperations
from query import QueryOperations


class ParquEdit:
    """A class for managing DuckDB tables with DuckLake catalog integration.

    This facade provides a unified interface to DDL, DML, and Query operations.
    
    For detailed documentation of each method, see the respective operations classes:
    - DDL operations: ddl.DDLOperations (create_table, drop_table, alter_table)
    - DML operations: dml.DMLOperations (insert, update, delete, fill_table)
    - Query operations: query.QueryOperations (view_table, select, count)
    
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
        ...     result = editor.view_table("users", limit=10, where="age > 25")
        ...     
        ...     # Update data
        ...     editor.update("users", {"status": "active"}, where="id = 1")
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
        table_description: str,
        part_columns: list[str] | None = None,
        fill: bool = False,
    ) -> None:
        """Create a new table. See DDLOperations.create_table for details."""
        self._ddl.create_table(table_name, source, table_description, part_columns, fill=False)
        if fill:
            self._dml.fill_table(table_name, source)
    
    def drop_table(self, table_name: str) -> None:
        """Drop a table. See DDLOperations.drop_table for details."""
        return self._ddl.drop_table(table_name)
    
    def alter_table(self, table_name: str, changes: dict[str, Any]) -> None:
        """Alter table structure. See DDLOperations.alter_table for details."""
        return self._ddl.alter_table(table_name, changes)
    
    # ============ DML Operations ============
    # Delegate to DMLOperations - see dml.py for full documentation
    
    def fill_table(
        self, table_name: str, source: pd.DataFrame | dict[str, Any] | str
    ) -> None:
        """Populate table with data. See DMLOperations.fill_table for details."""
        return self._dml.fill_table(table_name, source)
    
    def insert(self, table_name: str, data: pd.DataFrame) -> None:
        """Insert data into table. See DMLOperations.insert for details."""
        return self._dml.insert(table_name, data)
    
    def update(self, table_name: str, updates: dict[str, Any], where: str) -> None:
        """Update table rows. See DMLOperations.update for details."""
        return self._dml.update(table_name, updates, where)
    
    def delete(self, table_name: str, where: str) -> None:
        """Delete table rows. See DMLOperations.delete for details."""
        return self._dml.delete(table_name, where)
    
    # ============ Query Operations ============
    # Delegate to QueryOperations - see query.py for full documentation
    
    def view_table(
        self,
        table_name: str,
        limit: int | None = 10,
        offset: int = 0,
        columns: list[str] | None = None,
        where: str | None = None,
        order_by: str | None = None,
    ) -> pd.DataFrame:
        """View table contents. See QueryOperations.view_table for details."""
        return self._query.view_table(table_name, limit, offset, columns, where, order_by)
    
    def select(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Select data from table. See QueryOperations.select for details."""
        return self._query.select(table_name, columns, where, limit)
    
    def count(self, table_name: str, where: str | None = None) -> int:
        """Count table rows. See QueryOperations.count for details."""
        return self._query.count(table_name, where)
    
    def exists(self, table_name: str) -> bool:
        """Check if table exists. See QueryOperations.table_exists for details."""
        return self._query.table_exists(table_name)
