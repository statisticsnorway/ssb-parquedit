"""Local DuckDB connection backed by SQLite and local filesystem."""

import duckdb

from .connection import DuckDBConnection


class LocalDuckDBConnection(DuckDBConnection):
    """A real DuckDBConnection backed by DuckLake/SQLite for unit testing and local development.

    Bypasses DuckDBConnection.__init__ to avoid GCS and PostgreSQL
    dependencies, using a local SQLite catalog and a temporary directory for data files instead.
    """

    def __init__(self, data_path: str) -> None:
        """Create a DuckLake connection backed by SQLite at the given data_path.

        Args:
            data_path: Local directory used for both the SQLite catalog
                (catalog.db) and Parquet data files (data/).
        """
        self.data_path = data_path
        self._conn = duckdb.connect()
        self._conn.sql("INSTALL sqlite; LOAD sqlite;")
        self._conn.sql("INSTALL ducklake; LOAD ducklake;")
        self._conn.sql(f"""
            ATTACH 'ducklake:sqlite:{data_path}/catalog.db' AS test_catalog
            (DATA_PATH '{data_path}/data',
             DATA_INLINING_ROW_LIMIT 300)
        """)
        self._conn.sql("USE test_catalog")
