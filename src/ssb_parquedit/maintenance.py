"""Maintenance operations for Ducklake tables."""

import logging
import re
import shutil
from pathlib import Path
from typing import Any
from typing import cast

import gcsfs
import pandas as pd

from .local import LocalDuckDBConnection
from .utils import SchemaUtils

# Configure module-level logger
logger = logging.getLogger(__name__)


class MaintenanceOperations:
    """Maintenance operations for Ducklake tables.

    This class handles:
    - Flushing inlined data
    """

    def __init__(
        self, connection: Any, db_config: dict[str, str]  
    ) -> None:
        """Initialize with a DuckDB connection.

        Args:
            connection: DuckDBConnection instance.
            db_config: Optional database configuration dict for table cleanup operations.
                Required keys for cleanup: data_path, catalog_name.
        """
        self.conn = connection
        self.db_config: dict[str, str] | None = db_config

    def flush_inlined_table(self, table_name: str) -> None:
        """Flush inlined data for a table to Parquet storage.

        Args:
            table_name: Name of the table to flush.

        Raises:
            ValueError: If table_name is invalid.
            RuntimeError: If db_config is not initialized or flush fails.
        """
        try:
            SchemaUtils.validate_table_name(table_name)
        except ValueError as e:
            logger.error(str(e))
            raise

        if self.db_config is None:
            raise RuntimeError("db_config is not initialized")

        #self.conn.execute(f"CALL ducklake_flush_inlined_data({self.db_config["catalog_name"]}, schema_name => '{self.db_config["metadata_schema"]}', table_name => '{table_name}');")
        #res = self.conn.execute("SELECT schema_name, table_name, rows_flushed FROM ducklake_flush_inlined_data('dapla_ffunk')").df()
        #return res        
        
        self.conn.execute(f"CALL ducklake_flush_inlined_data({self.db_config["catalog_name"]}, table_name => '{table_name}');")

        #res = self.conn.execute(f"SELECT rows_flushed FROM ducklake_flush_inlined_data('dapla_ffunk', table_name => '{table_name}')").df()

        #rows_flushed = self.conn.execute(
        #        f"SELECT SUM(rows_flushed) FROM ducklake_flush_inlined_data('{self.db_config['catalog_name']}', "
        #        f"table_name => '{table_name}')"
        #    ).fetchone()[0]

        #logger.info(f"Flushed {rows_flushed} rows for {table_name}.")    

