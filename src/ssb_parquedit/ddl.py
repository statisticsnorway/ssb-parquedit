"""DDL (Data Definition Language) operations for DuckDB tables."""

from typing import Any
import duckdb

import pandas as pd
from utils import SchemaUtils
from utils import SQLSanitizer


class DDLOperations:
    """DDL operations for creating and modifying table structures.

    This class handles:
    - Table creation from DataFrames, schemas, or Parquet files
    - Table dropping and alteration
    - Partitioning configuration
    - Table descriptions/comments
    """

    def __init__(self, connection: duckdb.DuckDBConnection) -> None:
        """Initialize with a DuckDB connection.

        Args:
            connection: DuckDBConnection instance.
        """
        self.conn = connection

    def create_table(
        self,
        table_name: str,
        source: pd.DataFrame | dict[str, Any] | str,
        part_columns: list[str] | None = None,
    ) -> None:
        """Create a new table in the DuckLake catalog.

        Args:
            table_name: Name of the table to create. Must start with a letter or
                underscore and contain only alphanumeric characters and underscores.
            source: Source for table schema. Can be:
                - pd.DataFrame: Uses DataFrame schema to create table structure
                - dict: JSON Schema specification defining the table structure
                - str: Path to Parquet file (gs:// format) to infer schema from
            part_columns: List of column names to partition by. Defaults to None (no partitioning).
                When specified, the table will be partitioned by these columns for better query performance.
            fill: Whether to populate the table with data from source. Defaults to False.
                Note: This parameter is handled by the facade, not internally in this method.

        Raises:
            ValueError: If table_name contains invalid characters.
            TypeError: If source is not a DataFrame, dict, or string.

        Example:
            >>> # Create from DataFrame
            >>> df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
            >>> ddl.create_table("users", df, "User data")

            >>> # Create from JSON Schema
            >>> schema = {
            ...     "properties": {
            ...         "id": {"type": "integer"},
            ...         "name": {"type": "string"}
            ...     },
            ...     "required": ["id"]
            ... }
            >>> ddl.create_table("users", schema, "User data")

            >>> # Create from Parquet file with partitioning
            >>> ddl.create_table("events", "gs://bucket/events.parquet",
            ...                  "Event data", part_columns=["date", "region"])
        """
        SchemaUtils.validate_table_name(table_name)

        if isinstance(source, pd.DataFrame):
            self._create_from_dataframe(table_name, source)
        elif isinstance(source, dict):
            self._create_from_schema(table_name, source)
        elif isinstance(source, str):
            self._create_from_parquet(table_name, source)
        else:
            raise TypeError(
                "source must be a DataFrame, JSON Schema dict, or gs:// Parquet path"
            )

        if part_columns is None:
            part_columns = []
        if len(part_columns) > 0:
            self._add_table_partition(table_name, part_columns)

    def _create_from_dataframe(self, table_name: str, data: pd.DataFrame) -> None:
        """Create an empty table from a DataFrame schema.

        Args:
            table_name: Name of the table to create.
            data: DataFrame whose schema will be used.
        """
        cols = []

        # Stable UUID primary key
        cols.append("_id VARCHAR")

        for col, dtype in data.dtypes.items():
            duck_type = SchemaUtils.pandas_to_duckdb(dtype)
            cols.append(f"{col} {duck_type}")

        ddl = f"""
        CREATE TABLE {table_name} (
            {', '.join(cols)}
        );
        """

        self.conn.execute(ddl)

    def _create_from_parquet(self, table_name: str, parquet_path: str) -> None:
        """Create an empty table from a Parquet file schema.

        Args:
            table_name: Name of the table to create.
            parquet_path: Path to the Parquet file (supports gs:// URIs).
        """
        # Use parameterized query for the file path to prevent injection
        ddl = f"""
        CREATE TABLE {table_name} AS
        SELECT
            CAST(NULL AS VARCHAR) AS _id,
            *
        FROM read_parquet(?)
        WHERE 1 = 2
        """

        self.conn.execute(ddl, [parquet_path])

    def _create_from_schema(self, table_name: str, schema: dict[str, Any]) -> None:
        """Create a table from a JSON Schema specification.

        Args:
            table_name: Name of the table to create.
            schema: JSON Schema dictionary defining the table structure.
        """
        ddl = SchemaUtils.jsonschema_to_duckdb(schema, table_name)
        self.conn.execute(ddl)

    def _add_table_partition(self, table_name: str, part_columns: list[str]) -> None:
        """Configure partitioning for a table.

        Args:
            table_name: Name of the table to partition.
            part_columns: List of column names to partition by.

        Raises:
            SQLInjectionError: If any column name is invalid.
        """
        # Validate column names to prevent injection
        SQLSanitizer.validate_column_list(part_columns)

        cols = ", ".join(part_columns)
        self.conn.execute(f"ALTER TABLE {table_name} SET PARTITIONED BY ({cols});")
