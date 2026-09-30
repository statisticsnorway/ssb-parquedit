"""DDL (Data Definition Language) operations for DuckDB tables."""

import logging
import re
import shutil
from pathlib import Path
from typing import Any

import gcsfs
import pandas as pd
import polars as pl

from .local import LocalDuckDBConnection
from .utils import SchemaUtils

# Configure module-level logger
logger = logging.getLogger(__name__)


class DDLOperations:
    """DDL operations for creating and modifying table structures.

    This class handles:
    - Table creation from DataFrames, schemas, or Parquet files
    - Table dropping and alteration with snapshot expiration and storage cleanup
    - Partitioning configuration
    - Table descriptions/comments
    """

    def __init__(
        self, connection: Any, db_config: dict[str, str] | None = None
    ) -> None:
        """Initialize with a DuckDB connection.

        Args:
            connection: DuckDBConnection instance.
            db_config: Optional database configuration dict for table cleanup operations.
                Required keys for cleanup: data_path, catalog_name.
        """
        self.conn = connection
        self.db_config = db_config

    def create_table(
        self,
        table_name: str,
        source: Any,
        part_columns: list[str] | None = None,
    ) -> None:
        """Create a new table in the DuckLake catalog.

        Args:
            table_name: Name of the table to create. Must start with a letter or
                underscore and contain only alphanumeric characters and underscores.
            source: Source for table schema. Can be:
                - pd.DataFrame: Uses DataFrame schema to create table structure
                - pl.DataFrame: Uses DataFrame schema to create table structure
                - dict: JSON Schema specification defining the table structure
                - str: Path to Parquet file (gs:// format) to infer schema from
            part_columns: Optional list of column names to partition by.

        Raises:
            ValueError: If table_name contains invalid characters.
            TypeError: If source is not a DataFrame, dict, or string.

        Returns:
            None
        """
        # Make ValueError explicit for pydoclint (DOC503)
        try:
            SchemaUtils.validate_table_name(table_name)
        except ValueError as e:
            # Re-raise to make the exception visible to lint
            logger.error(str(e))
            raise

        if isinstance(source, dict):
            self._create_from_schema(table_name, source)
        elif isinstance(source, str):
            self._create_from_parquet(table_name, source)
        elif (
            SchemaUtils.is_dataframe(source) or source.__class__.__name__ == "DataFrame"
        ):
            self._create_from_dataframe(table_name, source)
        else:
            msg = "source must be a DataFrame, JSON Schema dict, or gs:// Parquet path"
            logger.error(msg)
            raise TypeError(msg)

        if part_columns is None:
            part_columns = []
        if len(part_columns) > 0:
            self._add_table_partition(table_name, part_columns)

        logger.info("Created table '%s'", table_name)

    def drop_table(self, table_name: str, cleanup: bool = False) -> None:
        """Drop a table from the DuckLake catalog.

        By default, only removes the table from the catalog. DuckLake preserves
        data files and snapshot history until snapshots are explicitly expired,
        so edit history remains accessible via snapshots() after a normal drop.

        When cleanup=True, additionally expires snapshots and deletes GCS data files.
        This permanently destroys all history and cannot be undone.

        Args:
            table_name: Name of the table to drop.
            cleanup: If True, expire snapshots and delete GCS data files after
                dropping. Defaults to False. History is permanently lost.

        Raises:
            ValueError: If table_name is invalid.

        Returns:
            None

        Example:
            >>> # doctest: +SKIP
            >>> ddl = DDLOperations(conn, db_config)
            >>> ddl.drop_table("my_table")           # History preserved
            >>> ddl.drop_table("my_table", cleanup=True)  # Full deletion
        """
        try:
            SchemaUtils.validate_table_name(table_name)
        except ValueError as e:
            logger.error(str(e))
            raise

        table_location = None
        if cleanup:
            try:
                table_location = self._get_table_location(table_name)
            except Exception as e:
                logger.warning(
                    "Could not retrieve table location for '%s': %s. "
                    "Proceeding with drop only, GCS files may need manual cleanup.",
                    table_name,
                    e,
                )

        self.conn.execute(f"DROP TABLE {table_name}")
        logger.warning("Dropped table: '%s'", table_name)

        if cleanup:
            self._expire_snapshots(table_name)
            if table_location:
                if isinstance(self.conn, LocalDuckDBConnection):
                    self._cleanup_local_files(table_location, table_name)
                else:
                    self._cleanup_gcs_files(table_location, table_name)

    def _get_table_location(self, table_name: str) -> str:
        """Get the storage location of a table.

        For local connections, returns ``~/.parquedit/data/main/<table>``.
        For remote (GCS) connections, returns ``{data_path}/{schema}/<table>``.

        Args:
            table_name: Name of the table.

        Returns:
            Path where the table data is stored.

        Raises:
            RuntimeError: If table location cannot be determined.
        """
        if isinstance(self.conn, LocalDuckDBConnection):
            return str(Path(self.conn.data_path) / "data" / "main" / table_name)

        if self.db_config and "data_path" in self.db_config:
            data_path = self.db_config["data_path"]
            try:
                row = self.conn.execute("SELECT CURRENT_SCHEMA()").fetchone()
                schema = row[0] if row and row[0] else "main"
            except Exception:
                schema = "main"
            return f"{data_path}/{schema}/{table_name}"

        msg = f"Cannot determine table location for {table_name}: no data_path configured."
        logger.error(msg)
        raise RuntimeError(msg)

    def _expire_snapshots(self, table_name: str) -> None:
        """Expire edit snapshots for a table, permanently removing its history.

        Queries snapshots() for all snapshot IDs associated with the given table
        (identified via commit_extra_info) and calls ducklake_expire_snapshots to
        mark them for deletion. Only called from drop_table(cleanup=True).

        Args:
            table_name: Name of the dropped table.
        """
        try:
            row = self.conn.execute("SELECT CURRENT_DATABASE()").fetchone()
            catalog_name = row[0] if row and row[0] else None
        except Exception:
            logger.exception("Could not determine current catalog for '%s'", table_name)
            return

        if not catalog_name:
            logger.warning(
                "Cannot expire snapshots for '%s': no active catalog.", table_name
            )
            return

        try:
            rows = self.conn.execute(
                "SELECT snapshot_id FROM snapshots() "
                "WHERE commit_extra_info IS NOT NULL "
                "AND json_extract_string(commit_extra_info, '$.table_name') = ?",
                [table_name],
            ).fetchall()

            snapshot_ids = [row[0] for row in rows]
            if not snapshot_ids:
                logger.info("No edit snapshots found to expire for '%s'.", table_name)
                return

            ids_literal = "[" + ", ".join(str(sid) for sid in snapshot_ids) + "]"
            self.conn.execute(
                f"CALL ducklake_expire_snapshots('{catalog_name}', versions => {ids_literal})"
            )
            logger.info(
                "Expired %d snapshot(s) for '%s'.", len(snapshot_ids), table_name
            )
        except Exception:
            logger.exception("Error during snapshot expiration for '%s'", table_name)

    def _cleanup_gcs_files(self, table_location: str, table_name: str) -> None:
        """Clean up orphaned files from GCS bucket.

        After dropping a Delta table, removes the data directory from GCS
        to reclaim storage space. This operation is safe because DuckLake
        uses metadata to track what files are part of the table.

        Args:
            table_location: GCS path to the table's data directory.
            table_name: Name of the dropped table (for logging).
        """
        try:
            # Parse GCS path
            match = re.match(r"gs://([^/]+)/(.+)", table_location)
            if not match:
                logger.error(
                    "Invalid GCS path format: %s. Cannot cleanup GCS files for '%s'. "
                    "Manual cleanup may be required.",
                    table_location,
                    table_name,
                )
                return

            # Use the full GCS path for deletion (as expected by tests)
            fs = gcsfs.GCSFileSystem()
            if fs.exists(table_location):
                logger.warning(
                    "Deleting table data from GCS: %s (table: '%s'). "
                    "This action cannot be undone.",
                    table_location,
                    table_name,
                )
                fs.rm(table_location, recursive=True)
                logger.info(
                    "Cleaned up GCS files for '%s' at %s", table_name, table_location
                )
            else:
                logger.warning(
                    "Table location not found in GCS: %s. "
                    "Data may only be inlined, already been deleted or path is incorrect.",
                    table_location,
                )
        except Exception as e:
            # GCS cleanup is optional - log warning but don't fail the drop operation
            logger.error(
                "Failed to clean up GCS files for '%s' at %s: %s. "
                "Files may need manual cleanup. Verify path and GCS permissions.",
                table_name,
                table_location,
                e,
            )

    def _cleanup_local_files(self, table_location: str, table_name: str) -> None:
        """Clean up orphaned files from the local filesystem.

        Local counterpart to ``_cleanup_gcs_files`` for ``LocalDuckDBConnection``.

        Args:
            table_location: Local path to the table's data directory.
            table_name: Name of the dropped table (for logging).
        """
        try:
            path = Path(table_location)
            if not path.exists():
                logger.warning(
                    "Table location not found locally: %s. "
                    "Data may have already been deleted or path is incorrect.",
                    table_location,
                )
                return

            logger.warning(
                "Deleting table data locally: %s (table: '%s'). "
                "This action cannot be undone.",
                table_location,
                table_name,
            )
            shutil.rmtree(path)
            logger.info(
                "Cleaned up local files for '%s' at %s", table_name, table_location
            )
        except Exception:
            logger.exception(
                "Failed to clean up local files for '%s' at %s. "
                "Files may need manual cleanup.",
                table_name,
                table_location,
            )

    def _create_from_dataframe(
        self, table_name: str, data: pd.DataFrame | pl.DataFrame
    ) -> None:
        """Create an empty table from a DataFrame schema.

        Args:
            table_name: Name of the table to create.
            data: DataFrame whose schema will be used.
        """
        SchemaUtils.validate_column_names(list(data.columns))
        if isinstance(data, pd.DataFrame):
            data = data.astype(
                {
                    col: object
                    for col, dtype in data.dtypes.items()
                    if isinstance(dtype, pd.StringDtype)
                }
            )
        self.conn.register("data", data)
        self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM data WHERE 1=2")

    def _create_from_parquet(self, table_name: str, parquet_path: str) -> None:
        """Create an empty table from a Parquet file schema.

        Args:
            table_name: Name of the table to create.
            parquet_path: Path to the Parquet file (supports gs:// URIs).
        """
        columns = self.conn.execute(
            "SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet(?))",
            [parquet_path],
        ).fetchall()
        SchemaUtils.validate_column_names([row[0] for row in columns])

        # Use parameterized query for the file path to prevent injection
        ddl = f"""
        CREATE TABLE {table_name} AS
        SELECT
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
        SchemaUtils.validate_column_names(list(schema["properties"].keys()))
        ddl = SchemaUtils.jsonschema_to_duckdb(schema, table_name)
        self.conn.execute(ddl)

    def _add_table_partition(self, table_name: str, part_columns: list[str]) -> None:
        """Configure partitioning for a table.

        Args:
            table_name: Name of the table to partition.
            part_columns: List of column names to partition by.
        """
        cols = ", ".join(part_columns)
        self.conn.execute(f"ALTER TABLE {table_name} SET PARTITIONED BY ({cols})")
        logger.info("Partitioned table '%s' by (%s)", table_name, cols)
