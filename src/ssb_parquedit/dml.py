"""DML (Data Manipulation Language) operations for DuckDB tables."""

import logging
import uuid
from typing import Any, Literal
import json
from datetime import datetime
import zoneinfo

import pandas as pd
import pyarrow as pa

from ssb_parquedit.functions import get_dapla_user

from .utils import SchemaUtils
from .query import QueryOperations

logger = logging.getLogger(__name__)

VALID_UPDATE_CAUSES = Literal["OTHER_SOURCE", "REVIEW", "OWNER", "MARGINAL_UNIT", "DUPLICATE", "OTHER"]

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
            msg = "source must be a DataFrame or gs:// Parquet path"
            logger.error(msg)
            raise TypeError(msg)

    def _insert_from_dataframe(self, table_name: str, data: pd.DataFrame) -> None:
        """Insert data from a DataFrame into a table.

        Args:
            table_name: Name of the table to populate.
            data: DataFrame containing the data to insert.
        """
        # Validate table name
        SchemaUtils.validate_table_name(table_name)

        df_copy = data.copy()

        df_copy.insert(0, "_id", [str(uuid.uuid4()) for _ in range(len(df_copy))])

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

        logger.debug("Inserting %d rows into '%s'", len(data), table_name)
        self.conn.execute(f"INSERT INTO {table_name} ({cols}) SELECT * FROM data")
        logger.debug("Insert complete: %d rows -> '%s'", len(data), table_name)

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

    def _validate_table_and_columns(self, table_name: str, changes: dict):
        # 1) fetch all table names and check
        valid_tables = {
            row[0] for row in self.conn.execute("""
                SELECT table_name
                FROM information_schema.tables
            """).fetchall()
        }
        if table_name not in valid_tables:
            raise ValueError(f"Table '{table_name}' does not exist")

        # 2) fetch all columns for table and check
        valid_columns = {
            row[0] for row in self.conn.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = ?
            """, [table_name]).fetchall()
        }
        missing = set(changes.keys()) - valid_columns
        if missing:
            raise ValueError(f"Missing columns in '{table_name}': {missing}")

    def edit(self, table_name: str, row_id: int, changes: dict[str, Any], change_event_reason: str, change_comment: str) -> int:
        # validate cause — specific to update
        if change_event_reason not in VALID_UPDATE_CAUSES.__args__:
            raise ValueError(f"Invalid cause: '{change_event_reason}'. Must be one of: {VALID_UPDATE_CAUSES.__args__}")

        # validate table and columns — shared
        self._validate_table_and_columns(table_name, changes)

        set_clause = ", ".join(f"{col} = ?" for col in changes.keys())
        values = list(changes.values()) + [row_id]

        try:
            self.conn.execute("BEGIN")

            #result = self.conn.execute(f"""
            #    UPDATE {table_name}
            #    SET {set_clause}
            #    WHERE rowid = ?
            #    RETURNING *
            #""", values).fetchall()

            #rows_affected = len(result)

            # safety net — rowid should always be unique
            #if rows_affected != 1:
            #    raise ValueError(f"Expected 1 row affected, got {rows_affected} for rowid {row_id} in '{table_name}'")

            self.conn.execute(f"""
            UPDATE {table_name}
            SET {set_clause}
            WHERE rowid = ?
            """, values)

            dapla_user = get_dapla_user()

            query = QueryOperations(self.conn)

            extra_info = json.dumps({
                "change_event_reason": change_event_reason,
                "changed_by": dapla_user,
                "change_comment": change_comment,
                "change_datetime": str(datetime.now(zoneinfo.ZoneInfo("Europe/Oslo"))),
                "statistics_name": query._get_product_name(table_name)
            })

            self.conn.execute("CALL set_commit_message(?, ?, ?)", [dapla_user, None, extra_info])

            self.conn.execute("COMMIT")

        except Exception as e:
            self.conn.execute("ROLLBACK")
            raise e

        #return rows_affected            