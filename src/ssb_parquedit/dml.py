"""DML (Data Manipulation Language) operations for DuckDB tables."""

import uuid
from typing import Any

import pandas as pd

from .utils import SchemaUtils


class DMLOperations:
    """DML operations for inserting, updating, and deleting table data.

    This class handles:
    - Data insertion from DataFrames or Parquet files
    - Row updates with filtering
    - Row deletions with filtering
    """

    # def __init__(self, connection) -> None:
    def __init__(self, connection: Any) -> None:
        """Initialize with a DuckDB connection.

        Args:
            connection: DuckDBConnection instance.
        """
        self.conn = connection

    def insert_data(self, table_name: str, source: Any) -> None:
        """Populate an existing table with data.

        Args:
            table_name: Name of the table to fill.
            source: Data source. Can be:
                - pd.DataFrame: Insert DataFrame rows into the table
                - str: Path to Parquet file (gs:// format) to read and insert data from

        Raises:
            TypeError: If source is not a DataFrame or string.

        Example:
            >>> # doctest: +SKIP
            >>> # Fill from DataFrame
            >>> df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
            >>> dml.fill_table("users", df)

            >>> # Fill from Parquet file
            >>> dml.fill_table("users", "gs://bucket/users.parquet")
        """
        if isinstance(source, pd.DataFrame):
            self._insert_from_dataframe(table_name, data=source)
        elif isinstance(source, str):
            self._insert_from_parquet(table_name, parquet_path=source)
        else:
            raise TypeError("source must be a DataFrame or gs:// Parquet path")

    def _insert_from_dataframe(self, table_name: str, data: pd.DataFrame) -> None:
        """Insert data from a DataFrame into a table.

        Args:
            table_name: Name of the table to populate.
            data: DataFrame containing the data to insert.
        """
        # Validate table name
        SchemaUtils.validate_table_name(table_name)

        df_copy = data.copy()

        # Insert _id as first column with string UUIDs
        df_copy.insert(0, "_id", [str(uuid.uuid4()) for _ in range(len(df_copy))])

        # Convert StringDtype columns to object dtype for broader compatibility
        if isinstance(df_copy, pd.DataFrame):
            df_copy = df_copy.astype(
                {
                    col: object
                    for col, dtype in df_copy.dtypes.items()
                    if isinstance(dtype, pd.StringDtype)
                }
            )

        self.conn.register("data", df_copy)

        # Insert into table - table name is validated, so safe to interpolate
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM data")

    def _insert_from_parquet(self, table_name: str, parquet_path: str) -> None:
        """Insert data from a Parquet file into a table.

        Args:
            table_name: Name of the table to populate.
            parquet_path: Path to the Parquet file (supports gs:// URIs).
        """
        # Validate table name
        SchemaUtils.validate_table_name(table_name)

        # Use parameterized query for the file path
        sql = f"""
        INSERT INTO {table_name}
        SELECT
            uuid()::VARCHAR AS _id,
            *
        FROM read_parquet(?)
        """

        self.conn.execute(sql, [parquet_path])
