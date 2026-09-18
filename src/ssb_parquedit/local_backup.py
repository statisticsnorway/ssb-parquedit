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
        """Attach a local DuckLake catalog file with GCS-hosted data.

        Reads DAPLA environment settings via ``create_config`` to determine the
        data path and metadata schema, then opens a DuckDB connection and
        attaches the local catalog file as a DuckLake catalog backed by data
        stored on GCS.

        Args:
            catalog_path: Filesystem path to the local DuckDB catalog file.
            catalog_name: Name to attach the catalog under. Defaults to
                ``"restored_catalog"`` when not given.
        """
        if catalog_name is None:
            catalog_name = "restored_catalog"

        self.db_config = create_config()
        self.catalog_path = catalog_path
        self.data_path = self.db_config["data_path"]
        self.catalog_name = catalog_name
        self.metadata_schema = self.db_config["metadata_schema"]
        self._conn = duckdb.connect()

        fs = gcsfs.GCSFileSystem()
        self._conn.register_filesystem(fs)
        self._conn.sql("INSTALL ducklake; LOAD ducklake;")

        self._conn.sql(f"""
            ATTACH 'ducklake:{self.catalog_path}' AS {self.catalog_name}
            (DATA_PATH '{self.data_path}',
             METADATA_SCHEMA '{self.metadata_schema}')
        """)
        self._conn.sql(f"USE {self.catalog_name}")
