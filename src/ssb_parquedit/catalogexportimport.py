"""Maintenance operations for Ducklake tables."""

import logging
from typing import Any
import datetime
from google.cloud import storage
import tempfile
import os

from .query import QueryOperations
from .maintenance import MaintenanceOperations

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

    def export_catalog(self) -> None:
        """Export metadata catalog to GCS.

        """
  
        if self.db_config is None:
            raise RuntimeError("db_config is not initialized")

        query = QueryOperations(self.conn, self.db_config)
        maintenance = MaintenanceOperations(self.conn, self.db_config)
        tables = query.list_tables()

        for table in tables:
            maintenance.flush_inlined_table(table)
            maintenance.merge_adjacent_files(table)
    
        client = storage.Client()

        schema = f"{self.db_config['metadata_schema']}"
        db = f"{self.db_config['dbname']}"
        user = f"{self.db_config['dbuser']}"
        data_path = f"{self.db_config['data_path']}"
        pg_connection_string = f"dbname={db} user={user} host=localhost"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file_name = f"{timestamp}_{schema}.duckdb"
        

        bucket = client.bucket(data_path.replace("/.parquedit_data", "").replace("gs://", ""))

        with tempfile.TemporaryDirectory() as tmp_dir:
            backup_file = os.path.join(tmp_dir, backup_file_name)      

            try:  
                self.conn.sql("BEGIN")

                self.conn.sql(f"ATTACH 'postgres:{pg_connection_string}' AS catalog_db (READ_ONLY);")
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

                self.conn.sql("DETACH catalog_db;")
                self.conn.sql("DETACH backup;") 

                self.conn.sql("COMMIT")

                blob = bucket.blob(f".parquedit_data/catalog-export/{backup_file_name}")
                blob.upload_from_filename(backup_file)
                print(f"Exported to: {data_path}/catalog-export/{backup_file_name}")          
                

            except Exception:
                try:
                    self.conn.sql("ROLLBACK")
                except Exception:
                    pass  # transaction already rolled back by DuckDB
                raise          