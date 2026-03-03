"""DDL (Data Definition Language) operations for DuckDB tables."""

from typing import Any

import pandas as pd

from .utils import SchemaUtils
from .utils import SQLInjectionError
from .utils import SQLSanitizer

class DDLOperations:
    """DDL operations for creating and modifying table structures.

    This class handles:
    - Table creation from DataFrames, schemas, or Parquet files
    - Table dropping and alteration
    - Partitioning configuration
    - Table descriptions/comments
    """

    def __init__(self, connection: Any) -> None:
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
            part_columns: Optional list of column names to partition by.

        Raises:
            ValueError: If table_name contains invalid characters.
            TypeError: If source is not a DataFrame, dict, or string.

        Returns:
            None
        """
        # Gjør ValueError eksplisitt for pydoclint (DOC503)
        try:
            SchemaUtils.validate_table_name(table_name)
        except ValueError as e:
            # Re-raise for å gjøre unntaket synlig for lint
            raise ValueError(str(e)) from e

        # Check if source is a DataFrame (handle both real and mock pandas DataFrames)
        if isinstance(source, dict):
            self._create_from_schema(table_name, source)
        elif isinstance(source, str):
            self._create_from_parquet(table_name, source)
        elif (
            isinstance(source, pd.DataFrame) or source.__class__.__name__ == "DataFrame"
        ):
            self._create_from_dataframe(table_name, source)
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
        # Register the DataFrame with DuckDB
        self.conn.register("_temp_df", data)

        # Create an empty table with the schema from the DataFrame
        ddl = f"""
        CREATE TABLE {table_name} AS
        SELECT
            CAST(NULL AS VARCHAR) AS _id,
            *
        FROM _temp_df
        WHERE 1 = 2
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
        try:
            SQLSanitizer.validate_column_list(part_columns)
        except SQLInjectionError as e:  
            # Re-raise for å gjøre unntaket synlig for linter
            raise SQLInjectionError(str(e)) from e

        cols = ", ".join(part_columns)
        self.conn.execute(f"ALTER TABLE {table_name} SET PARTITIONED BY ({cols});")
