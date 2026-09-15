"""Local DuckDB connection backed by DuckDB and GCS."""

import duckdb
import gcsfs

from .connection import DuckDBConnection
from .functions import create_config

class LocalCatalogGCSDataConnection(DuckDBConnection):
    """DuckDBConnection backed by a local DuckDB catalog file + GCS data."""

    def __init__(
        self,
        catalog_path: str,
        catalog_name: str | None = None,
             
    ) -> None:
        if catalog_name is None:
            catalog_name = "restored_catalog"
        
        self.db_config =  create_config()
        self.catalog_path = catalog_path
        self.data_path = self.db_config["data_path"]
        self.catalog_name = catalog_name
        self.metadata_schema = self.db_config["metadata_schema"]
        self._conn = duckdb.connect()

        print(self.db_config["data_path"])
        print(self.db_config["metadata_schema"])
        print(catalog_name)
        print(catalog_path)

        fs = gcsfs.GCSFileSystem()
        self._conn.register_filesystem(fs)
        self._conn.sql("INSTALL ducklake; LOAD ducklake;")
        
        self._conn.sql(f"""
            ATTACH 'ducklake:{self.catalog_path}' AS {self.catalog_name}
            (DATA_PATH '{self.data_path}',
             METADATA_SCHEMA '{self.metadata_schema}')
        """)
        self._conn.sql(f"USE {self.catalog_name}")