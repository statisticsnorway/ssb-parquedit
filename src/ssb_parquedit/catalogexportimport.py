"""Maintenance operations for Ducklake tables."""

import datetime
import logging
import os
import tempfile
from typing import Any

import gcsfs

from .maintenance import MaintenanceOperations
from .query import QueryOperations

logger = logging.getLogger(__name__)


class CatalogExportImport:
    """Catalog Export and Import.

    This class handles:
    - Exporting catalog
    """

    def __init__(self, connection: Any, db_config: dict[str, str]) -> None:
        """Initialize with a DuckDB connection.

        Args:
            connection: DuckDBConnection instance.
            db_config: Database configuration dict. Required key: catalog_name.
        """
        self.conn = connection
        self.db_config: dict[str, str] | None = db_config

    def export_catalog(self, export_path: str | None = None) -> str:
        """Export the DuckLake metadata catalog to GCS as a DuckDB backup file.

        Flushes and merges inlined data for every table in the catalog, then
        copies all tables from the PostgreSQL-backed catalog schema into a
        local DuckDB file, which is uploaded to GCS.

        Args:
            export_path: GCS path (without filename) to upload the backup to.
                Defaults to ``"{data_path}/catalog-export"`` when not given.

        Returns:
            The full GCS path (including filename) of the exported backup file.

        Raises:
            RuntimeError: If ``db_config`` is not initialized.
            Exception: If the export fails, the transaction is rolled back and
                the original exception is re-raised.
        """
        if self.db_config is None:
            raise RuntimeError("db_config is not initialized")

        if export_path is None:
            export_path = f"{self.db_config['data_path']}/catalog-export"

        query = QueryOperations(self.conn, self.db_config)
        maintenance = MaintenanceOperations(self.conn, self.db_config)
        tables = query.list_tables()

        for table in tables:
            maintenance.flush_inlined_table(table)
            maintenance.merge_adjacent_files(table)

        schema = f"{self.db_config['metadata_schema']}"
        db = f"{self.db_config['dbname']}"
        user = f"{self.db_config['dbuser']}"
        data_path = f"{self.db_config['data_path']}"
        pg_connection_string = f"dbname={db} user={user} host=localhost port={self.db_config['port_number']}"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file_name = f"{timestamp}_{schema}.duckdb"

        with tempfile.TemporaryDirectory() as tmp_dir:
            backup_file = os.path.join(tmp_dir, backup_file_name)

            try:
                self.conn.sql("BEGIN")

                self.conn.sql(
                    f"ATTACH 'postgres:{pg_connection_string}' AS catalog_db (READ_ONLY);"
                )
                self.conn.sql(f"ATTACH 'duckdb:{backup_file}' AS backup;")

                self.conn.sql(f"CREATE SCHEMA IF NOT EXISTS backup.{schema};")

                tables = self.conn.sql(f"""
                    SELECT table_name
                    FROM catalog_db.information_schema.tables
                    WHERE table_schema = '{schema}'
                """).fetchall()

                for (table_name,) in tables:
                    print(f"Copying {schema}.{table_name} ...")
                    self.conn.sql(f"""
                        CREATE OR REPLACE TABLE backup.{schema}.{table_name} AS
                        SELECT * FROM catalog_db.{schema}.{table_name}
                    """)
                print("Backup complete.")

                self.conn.sql("COMMIT")

                self.conn.sql("DETACH catalog_db;")
                self.conn.sql("DETACH backup;")

                fs = gcsfs.GCSFileSystem()
                fs.put(backup_file, f"{export_path}/{backup_file_name}")

                print(f"Exported to: {data_path}/catalog-export/{backup_file_name}")

            except Exception:
                try:
                    self.conn.sql("ROLLBACK")
                except Exception:
                    pass  # transaction already rolled back by DuckDB
                raise

        return f"{data_path}/catalog-export/{backup_file_name}"

    def import_catalog(self, backup_file_path: str) -> str:

        if self.db_config is None:
            raise RuntimeError("db_config is not initialized")

        schema = f"{self.db_config['metadata_schema']}"
        db = f"{self.db_config['dbname']}"
        user = f"{self.db_config['dbuser']}"
        pg_connection_string = f"dbname={db} user={user} host=localhost port={self.db_config['port_number']}"

        try:
            self.conn.sql("BEGIN")

            self.conn.sql(f"ATTACH 'postgres:{pg_connection_string}' AS restore_db;")
            self.conn.sql(f"ATTACH 'duckdb:{backup_file_path}' AS from_backup;")

            backup_tables = self.conn.sql("""
                SELECT table_name
                FROM duckdb_tables()
                WHERE database_name = 'from_backup'
            """).fetchall()

            for (table_name,) in backup_tables:
                print(f"Copying {schema}.{table_name} ...")

                self.conn.sql(f"""
                    DELETE FROM restore_db.{schema}.{table_name}
                """)
                self.conn.sql(f"""
                    INSERT INTO restore_db.{schema}.{table_name}
                    SELECT * FROM from_backup.{schema}.{table_name}
                """)

            print("Restore complete.")

            self.conn.sql("COMMIT")

            self.conn.sql("DETACH from_backup;")
            self.conn.sql("DETACH restore_db;")

        except Exception:
            try:
                self.conn.sql("ROLLBACK")
            except Exception:
                pass  # transaction already rolled back by DuckDB
            raise
