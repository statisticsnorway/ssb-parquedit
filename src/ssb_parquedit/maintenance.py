"""Maintenance operations for Ducklake tables."""

import logging
from typing import Any

from .utils import SchemaUtils

logger = logging.getLogger(__name__)


class MaintenanceOperations:
    """Maintenance operations for Ducklake tables.

    This class handles:
    - Flushing inlined data
    """

    def __init__(self, connection: Any, db_config: dict[str, str]) -> None:
        """Initialize with a DuckDB connection.

        Args:
            connection: DuckDBConnection instance.
            db_config: Database configuration dict. Required key: catalog_name.
        """
        self.conn = connection
        self.db_config: dict[str, str] | None = db_config

    def flush_inlined_table(self, table_name: str) -> None:
        """Flush inlined data for a table to Parquet storage.

        Args:
            table_name: Name of the table to flush.

        Raises:
            ValueError: If table_name is invalid.
            RuntimeError: If db_config is not initialized.
        """
        try:
            SchemaUtils.validate_table_name(table_name)
        except ValueError as e:
            logger.error(str(e))
            raise

        if self.db_config is None:
            raise RuntimeError("db_config is not initialized")

        catalog = self.db_config["catalog_name"]
        rows = self.conn.execute(
            "SELECT schema_name, table_name, rows_flushed "
            f"FROM ducklake_flush_inlined_data('{catalog}', table_name => '{table_name}')"
        ).fetchall()

        rows_flushed = sum(row[2] for row in rows)
        if rows:
            logger.info("Flushed %d rows for table '%s'.", rows_flushed, table_name)
        else:
            logger.info("No inlined data to flush for table '%s'.", table_name)
