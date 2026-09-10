"""Local DuckDB connection backed by DuckDB and GCS."""

import duckdb

from .connection import DuckDBConnection

import gcsfs

class LocalBackupDuckDBConnection(DuckDBConnection):
    """A real DuckDBConnection backed by DuckLake/DuckDB.

    Bypasses DuckDBConnection.__init__ to avoid  PostgreSQL
    dependencies, using a local DuckDB-backup catalog .
    """

    def __init__(self, data_path: str) -> None:
        """Create a DuckLake connection backed by local DuckDB Backup at the given data_path.

        Args:
            data_path: GCS-directory for Parquet data files.
        """
   
        self._conn = duckdb.connect()
        fs = gcsfs.GCSFileSystem()
        self._conn.register_filesystem(fs)     
        self._conn.sql("INSTALL ducklake; LOAD ducklake;")   

        #self._conn.sql(f"""
        #    ATTACH 'ducklake:duckdb:database.duckdb' AS my_duckdb_backup
        #    (DATA_PATH '{data_path}/data',
        #     DATA_INLINING_ROW_LIMIT 300);
        #    """) 
        #        
        
        self._conn.sql("""
            ATTACH 'ducklake:duckdb:database.duckdb' AS my_duckdb_backup
            (DATA_PATH 'gs://ssb-dapla-ffunk-data-produkt-prod/.parquedit_data');
            """)  
        
        self._conn.sql("USE my_duckdb_backup")