"""DDL (Data Definition Language) operations for DuckDB tables."""

import logging
import re
from typing import Any

import gcsfs
import pandas as pd

from .functions import get_dapla_environment
from .utils import SchemaUtils
from .utils import SQLInjectionError
from .utils import SQLSanitizer

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

    def drop_table(self, table_name: str, cleanup: bool = True) -> None:
        """Drop a table from the DuckLake catalog with optional cleanup.

        Table deletion is only allowed in the TEST environment to prevent
        accidental data loss in production. In PROD or other environments,
        this method will raise a PermissionError.

        Optionally performs comprehensive cleanup:
        - Expires snapshots (removes old transaction logs from metadata)
        - Cleans GCS bucket (removes orphaned Parquet files)

        Args:
            table_name: Name of the table to drop.
            cleanup: If True, expire snapshots and clean GCS files.
                Defaults to True. Requires db_config to be set.

        Raises:
            PermissionError: If DAPLA_ENVIRONMENT is not "test".
            ValueError: If table_name is invalid.

        Returns:
            None

        Example:
            >>> # doctest: +SKIP
            >>> ddl = DDLOperations(conn, db_config)
            >>> ddl.drop_table("temporary_table")  # Includes cleanup
            >>> ddl.drop_table("temp_table", cleanup=False)  # Drop only
        """
        # Validate table name
        try:
            SchemaUtils.validate_table_name(table_name)
        except ValueError as e:
            raise ValueError(str(e)) from e

        # Check environment
        environment = get_dapla_environment()
        if environment != "test":
            raise PermissionError(
                f"Table deletion is only allowed in TEST environment. "
                f"Current environment: {environment or 'not set'}. "
                f"Set DAPLA_ENVIRONMENT=test to enable table deletion."
            )

        # Get table location before dropping (if cleanup is enabled)
        table_location = None
        if cleanup:
            try:
                table_location = self._get_table_location(table_name)
            except Exception as e:
                logger.warning(
                    f"Could not retrieve table location for {table_name}: {e}. "
                    f"Proceeding with drop only."
                )
                cleanup = False

        # Execute drop
        self.conn.execute(f"DROP TABLE {table_name}")
        logger.warning(
            f"Dropped table: {table_name} from {environment.upper()} environment"
        )

        # Perform cleanup if enabled and location was retrieved
        if cleanup and table_location:
            self._expire_snapshots(table_name)
            self._cleanup_gcs_files(table_location, table_name)

    def _get_table_location(self, table_name: str) -> str:
        """Get the GCS location of a table.

        Args:
            table_name: Name of the table.

        Returns:
            GCS path where the table data is stored.

        Raises:
            RuntimeError: If table location cannot be determined.
        """
        try:
            result = self.conn.execute(
                "SELECT location FROM information_schema.tables WHERE table_name = ?"
                " AND table_schema = CURRENT_SCHEMA()",
                [table_name],
            )
            rows = result.fetchall()
            if rows and rows[0][0]:
                return str(rows[0][0])
        except Exception as e:
            logger.debug(f"Could not query table location: {e}")

        # Fallback: construct path from data_path and table_name
        if self.db_config and "data_path" in self.db_config:
            data_path = self.db_config["data_path"]
            # Typical DuckLake/Delta path structure
            return f"{data_path}/{table_name}"

        raise RuntimeError("Cannot determine table location for cleanup")

    def _expire_snapshots(self, table_name: str) -> None:
        """Expire old snapshots for a dropped table metadata cleanup.

        DuckLake stores transaction logs in metadata. This removes old
        snapshots to free up metadata storage space.

        Args:
            table_name: Name of the dropped table.
        """
        try:
            # DuckLake/Delta Lake snapshot expiration
            # Note: After DROP TABLE, the table is gone from catalog but metadata files remain
            self.conn.execute(
                "CALL delta_catalog.expire_snapshots("
                "schema_name => CURRENT_SCHEMA(), "
                "table_name => ?, "
                "older_than => CURRENT_TIMESTAMP - INTERVAL 7 DAY"
                ")",
                [table_name],
            )
            logger.info(f"Expired snapshots for {table_name}")
        except Exception as e:
            # Snapshot expiration is optional - log but don't fail
            logger.debug(f"Could not expire snapshots for {table_name}: {e}")

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
                logger.warning(f"Invalid GCS path format: {table_location}")
                return

            _bucket_name, _path_prefix = match.groups()

            # Initialize GCS filesystem
            fs = gcsfs.GCSFileSystem()

            # Remove the table's directory and all files within it
            if fs.exists(table_location):
                fs.rm(table_location, recursive=True)
                logger.info(
                    f"Cleaned up GCS files for {table_name} at {table_location}"
                )
            else:
                logger.debug(f"Table location not found in GCS: {table_location}")

        except Exception as e:
            # GCS cleanup is optional - log but don't fail the drop operation
            logger.warning(
                f"Could not clean up GCS files for {table_name} at "
                f"{table_location}: {e}. Files may need manual cleanup."
            )

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
