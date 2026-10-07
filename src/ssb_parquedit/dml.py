"""DML (Data Manipulation Language) operations for DuckDB tables."""

import json
import logging
from typing import Any
from typing import Literal
from typing import get_args

import pandas as pd
import polars as pl
import pyarrow as pa
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_not_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_random

from ssb_parquedit.functions import get_dapla_user

from .query import QueryOperations
from .utils import NestedPathUtils
from .utils import SchemaUtils

logger = logging.getLogger(__name__)

VALID_UPDATE_CAUSES = Literal[
    "OTHER_SOURCE", "REVIEW", "OWNER", "MARGINAL_UNIT", "DUPLICATE", "OTHER"
]


class DMLOperations:
    """DML operations for inserting, updating, and deleting table data.

    This class handles:
    - Data insertion from DataFrames or Parquet files
    - Row updates with filtering
    - Row deletions with filtering
    """

    # def __init__(self, connection) -> None:
    def __init__(
        self, connection: Any, db_config: dict[str, str] | None = None
    ) -> None:
        """Initialize with a DuckDB connection.

        Args:
            connection: DuckDBConnection instance.
            db_config: Optional database configuration dict. If None, defaults to an empty dict.
        """
        self.conn = connection
        self.db_config = db_config or {}

    def insert_data(self, table_name: str, source: Any) -> None:
        """Populate an existing table with data.

        Args:
            table_name: Name of the table to fill.
            source: Data source. Can be:
                - pd.DataFrame: Insert DataFrame rows into the table
                - pl.DataFrame: Insert DataFrame rows into the table
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
        if SchemaUtils.is_dataframe(source):
            self._insert_from_dataframe(table_name, data=source)
        elif isinstance(source, str):
            self._insert_from_parquet(table_name, parquet_path=source)
        else:
            msg = "source must be a DataFrame or gs:// Parquet path"
            logger.error(msg)
            raise TypeError(msg)

    def _insert_from_dataframe(
        self, table_name: str, data: pd.DataFrame | pl.DataFrame
    ) -> None:
        """Insert data from a DataFrame into a table.

        Args:
            table_name: Name of the table to populate.
            data: Pandas or polars DataFrame containing the data to insert.
        """
        # Validate table name
        SchemaUtils.validate_table_name(table_name)

        # target table's column types by name,
        col_types = {
            row[0]: row[1]
            for row in self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        }

        if isinstance(data, pd.DataFrame):
            arrow_table = self._pandas_to_arrow(data, col_types)
        else:
            arrow_table = self._polars_to_arrow(data, col_types)

        self.conn.register("data", arrow_table)

        cols = ", ".join(arrow_table.schema.names)

        logger.debug("Inserting %d rows into '%s'", len(data), table_name)
        self.conn.execute(f"INSERT INTO {table_name} ({cols}) SELECT * FROM data")
        logger.info("Inserted %d row(s) into '%s'", len(data), table_name)

    @staticmethod
    def _pandas_to_arrow(data: pd.DataFrame, col_types: dict[str, str]) -> pa.Table:
        df_copy = data.copy()

        # Keep target-compatible types before Arrow infers the table schema.
        for col in df_copy.columns:
            if col_types.get(col) == "VARCHAR":
                df_copy[col] = df_copy[col].astype(str)
            elif col_types.get(col) == "BIGINT":
                df_copy[col] = pd.Series(
                    pd.to_numeric(df_copy[col], errors="coerce"), dtype="Int64"
                )

        return pa.Table.from_pandas(df_copy, preserve_index=False)

    @staticmethod
    def _polars_to_arrow(data: pl.DataFrame, col_types: dict[str, str]) -> pa.Table:
        cast_exprs = []
        for col in data.columns:
            if col_types.get(col) == "VARCHAR":
                cast_exprs.append(pl.col(col).cast(pl.Utf8))
            elif col_types.get(col) == "BIGINT":
                cast_exprs.append(pl.col(col).cast(pl.Int64, strict=False))

        if cast_exprs:
            data = data.with_columns(cast_exprs)
        return data.to_arrow()

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
            *
        FROM read_parquet(?)
        """

        self.conn.execute(sql, [parquet_path])
        logger.info(
            "Inserted data into '%s' from Parquet file '%s'", table_name, parquet_path
        )

    def _validate_table_and_columns(
        self, table_name: str, changes: dict[str, Any]
    ) -> None:
        # 1) fetch all table names and check
        valid_tables = {row[0] for row in self.conn.execute("""
                SELECT table_name
                FROM information_schema.tables
            """).fetchall()}
        if table_name not in valid_tables:
            msg = f"Table '{table_name}' does not exist"
            logger.error(msg)
            raise TypeError(msg)

        # 2) fetch all columns for table and check
        valid_columns = {
            row[0]
            for row in self.conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = ?
            """,
                [table_name],
            ).fetchall()
        }
        # Keys may be plain column names or nested paths into a STRUCT/LIST
        # column (e.g. "address.city", "items[1].qty") — only the base
        # column needs to exist in the catalog.
        base_columns = {NestedPathUtils.parse_path(key)[0] for key in changes.keys()}
        missing = base_columns - valid_columns
        if missing:
            msg = f"Missing columns in '{table_name}': {missing}"
            logger.error(msg)
            raise TypeError(msg)

    @retry(
        stop=stop_after_attempt(max_attempt_number=10),
        wait=wait_random(min=1, max=3),
        retry=retry_if_not_exception_type((ValueError, TypeError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def edit(
        self,
        table_name: str,
        rowid: int,
        changes: dict[str, Any],
        change_event_reason: str,
        change_comment: str,
    ) -> None:
        """Edit a single row in a table by its row ID.

        Updates the specified columns for the row matching the given rowid.
        The change is wrapped in a transaction and committed with metadata
        including the change reason, comment, user, and timestamp.

        A key in `changes` may also be a nested path into a STRUCT or LIST
        column, using dotted/bracket notation, e.g. "address.city" (a field
        inside the STRUCT column "address") or "items[1].qty" (a field of
        element 1 of the LIST column "items"). Only the targeted nested
        field/element is replaced — the rest of the struct/list value is left
        untouched — and only that specific entry (not the whole column) is
        recorded in the changelog's old_values/new_values.

        Args:
            table_name: The name of the table to edit.
            rowid: The rowid of the row to update.
            changes: A dictionary mapping column names (or nested paths, see
                above) to their new values.
            change_event_reason: A reason code describing the type of change. Must be one of the valid update causes defined in VALID_UPDATE_CAUSES.
            change_comment: A human-readable comment describing the change.

        Raises:
            ValueError: If change_event_reason is not a valid update cause.
            Exception: Re-raises any exception that occurs during the transaction after rolling back.

        Example:
            >>> # doctest: +SKIP
            >>> # Replace an entire column
            >>> dml.edit("cities", 1, {"population": 650000}, "REVIEW", "Census update")
            >>> # Edit a single field inside a STRUCT column "address"
            >>> dml.edit("cities", 1, {"address.city": "Oslo"}, "REVIEW", "Typo fix")
            >>> # Edit a single element inside a LIST column "tags"
            >>> dml.edit("cities", 1, {"tags[0]": "capital"}, "REVIEW", "Tag fix")
            >>> # Edit a field of a struct inside a LIST column "items"
            >>> dml.edit("cities", 1, {"items[1].qty": 5}, "REVIEW", "Quantity fix")
        """
        # Coerce numpy/pandas integer types (e.g. numpy.int64 from a DataFrame
        # column) to a native Python int — DuckDB's parameter binding can't
        # handle numpy scalar types directly.
        rowid = int(rowid)

        # validate cause — specific to update
        if change_event_reason not in get_args(VALID_UPDATE_CAUSES):
            msg = f"Invalid cause: '{change_event_reason}'. Must be one of: {get_args(VALID_UPDATE_CAUSES)}"
            logger.error(msg)
            raise ValueError(msg)

        # validate table and columns — shared
        self._validate_table_and_columns(table_name, changes)

        # Parse each key into its base column and (possibly empty) nested
        # path, preserving the order in which nested edits were requested so
        # that multiple edits targeting the same STRUCT/LIST column are
        # applied one after another onto that column's value.
        parsed_changes = {
            key: NestedPathUtils.parse_path(key) for key in changes.keys()
        }

        dapla_user = get_dapla_user()

        query = QueryOperations(self.conn, self.db_config)

        # get product_name
        tag_dict = query._get_tag_info(table_name)
        if tag_dict is None:
            return
        product_name = tag_dict.get("product_name")

        # get user_defined_id
        user_defined_id = tag_dict.get("user_defined_id")

        try:
            self.conn.execute("BEGIN")

            # get current row
            row = self.conn.execute(
                f"SELECT * FROM {table_name} WHERE rowid = ?", [rowid]
            ).df()

            # make dict with values of unique_id-cols from row
            assert user_defined_id is not None
            unique_row = row[user_defined_id].iloc[0]
            key_values = {
                col: NestedPathUtils.to_native(val)
                for col, val in zip(user_defined_id, unique_row, strict=True)
            }

            # Build the actual column-level values to SET, merging nested
            # struct/list edits into their base column's current value, while
            # recording only the specific changed entry (not the whole
            # column) in old_values/new_values for the changelog.
            old_values: dict[str, Any] = {}
            new_values: dict[str, Any] = {}
            column_updates: dict[str, Any] = {}

            for key, (base_col, tokens) in parsed_changes.items():
                new_val = changes[key]
                if not tokens:
                    # Plain column edit — replace the whole value.
                    old_values[key] = NestedPathUtils.to_native(row[base_col].iloc[0])
                    column_updates[base_col] = new_val
                else:
                    # Nested edit — start from the column's current value (or
                    # from a prior nested edit already applied to it earlier
                    # in this same call) and replace only the targeted entry.
                    current = NestedPathUtils.to_native(
                        column_updates.get(base_col, row[base_col].iloc[0])
                    )
                    old_values[key] = NestedPathUtils.get_nested(current, tokens)
                    column_updates[base_col] = NestedPathUtils.set_nested(
                        current, tokens, new_val
                    )
                new_values[key] = new_val

            set_clause = ", ".join(f"{col} = ?" for col in column_updates.keys())
            values = [*column_updates.values(), rowid]

            extra_info = json.dumps(
                {
                    "change_type": "UPDATE",
                    "change_event_reason": change_event_reason,
                    "changed_by": dapla_user,
                    "table_name": table_name,
                    "rowid": rowid,
                    "affected_rows": 1,
                    "user_defined_id": key_values,
                    "change_comment": change_comment,
                    "product_name": product_name,
                    "old_values": old_values,
                    "new_values": new_values,
                }
            )

            self.conn.execute(
                f"""
            UPDATE {table_name}
            SET {set_clause}
            WHERE rowid = ?
            """,
                values,
            )

            self.conn.execute(
                "CALL set_commit_message(?, ?, ?)", [dapla_user, None, extra_info]
            )

            self.conn.execute("COMMIT")
            logger.info(
                "Edited row %d in '%s' (reason=%s)",
                rowid,
                table_name,
                change_event_reason,
            )

        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass  # transaction already rolled back by DuckDB
            raise

    @retry(
        stop=stop_after_attempt(max_attempt_number=10),
        wait=wait_random(min=1, max=3),
        retry=retry_if_not_exception_type((ValueError, TypeError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def delete_row(
        self,
        table_name: str,
        where: str,
        change_event_reason: str,
        change_comment: str,
    ) -> None:
        """Delete one or more rows from a table matching a WHERE clause.

        Selects the rows to delete using the same ``where`` filter syntax as
        ``QueryOperations.view()``, then deletes all matching rows in a single
        ``DELETE`` statement wrapped in one transaction. The whole batch is
        logged as a single changelog entry — recording how many rows were
        deleted and the ``user_defined_id`` of each — rather than one entry
        per row.

        Args:
            table_name: The name of the table to delete rows from.
            where: SQL WHERE clause (without the WHERE keyword) selecting the
                rows to delete, e.g. "population < 100000" or "id IN (1, 2)".
            change_event_reason: A reason code describing the type of change. Must be one of the valid update causes defined in VALID_UPDATE_CAUSES.
            change_comment: A human-readable comment describing the change.

        Raises:
            ValueError: If change_event_reason is not a valid update cause, or
                if no rows match the given where clause.
            Exception: Re-raised if the delete transaction fails for any other
                reason, after rolling back.
        """
        # validate cause — specific to update/delete
        if change_event_reason not in get_args(VALID_UPDATE_CAUSES):
            msg = f"Invalid cause: '{change_event_reason}'. Must be one of: {get_args(VALID_UPDATE_CAUSES)}"
            logger.error(msg)
            raise ValueError(msg)

        # validate table exists — shared
        self._validate_table_and_columns(table_name, {})

        dapla_user = get_dapla_user()

        query = QueryOperations(self.conn, self.db_config)

        # get product_name
        tag_dict = query._get_tag_info(table_name)
        if tag_dict is None:
            return
        product_name = tag_dict.get("product_name")

        try:
            self.conn.execute("BEGIN")

            # select the rows to delete, using the same `where` filter as view()
            matches = self.conn.execute(
                f"SELECT rowid FROM {table_name} WHERE {where}"
            ).df()

            if matches.empty:
                msg = f"No rows in table '{table_name}' match where clause: {where}"
                logger.error(msg)
                raise ValueError(msg)

            extra_info = json.dumps(
                {
                    "change_type": "DELETE",
                    "change_event_reason": change_event_reason,
                    "changed_by": dapla_user,
                    "table_name": table_name,
                    "where_clause": where,
                    "affected_rows": len(matches),
                    "user_defined_id": None,
                    "change_comment": change_comment,
                    "product_name": product_name,
                    "old_values": None,
                    "new_values": None,
                }
            )

            self.conn.execute(f"DELETE FROM {table_name} WHERE {where}")

            self.conn.execute(
                "CALL set_commit_message(?, ?, ?)", [dapla_user, None, extra_info]
            )

            self.conn.execute("COMMIT")
            logger.info(
                "Deleted %d row(s) from '%s' (reason=%s)",
                len(matches),
                table_name,
                change_event_reason,
            )

        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass  # transaction already rolled back by DuckDB
            raise
