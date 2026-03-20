"""DML (Data Manipulation Language) operations for DuckDB tables."""

from typing import Any

import pandas as pd
import pyarrow as pa

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

        # target table's column types by name,
        col_types = {
            row[0]: row[1]
            for row in self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        }

        # force the pandas column to pure Python str, Arrow will infer int64 if early values look numeric
        for col in df_copy.columns:
            if col_types.get(col) == "VARCHAR":
                df_copy[col] = df_copy[col].astype(str)
            elif col_types.get(col) == "BIGINT":
                df_copy[col] = pd.Series(
                    pd.to_numeric(df_copy[col], errors="coerce"), dtype="Int64"
                )

        arrow_table = pa.Table.from_pandas(df_copy, preserve_index=False)
        self.conn.register("data", arrow_table)

        cols = ", ".join(arrow_table.schema.names)
        self.conn.execute(f"INSERT INTO {table_name} ({cols}) SELECT * FROM data")

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
