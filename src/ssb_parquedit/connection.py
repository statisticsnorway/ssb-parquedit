"""DuckDB connection management with DuckLake catalog support."""

from typing import Any

import duckdb
import gcsfs


class DuckDBConnection:
    """Manages DuckDB connection with DuckLake catalog integration.

    This class handles:
    - DuckDB connection lifecycle
    - GCS filesystem registration
    - DuckLake and Postgres extension loading
    - Catalog attachment and configuration
    """

    def __init__(
        self, db_config: dict[str, str], conn: duckdb.DuckDBPyConnection | None = None
    ) -> None:
        """Initialize DuckDB connection with catalog.

        Args:
            db_config: Database configuration dict with keys:
                - dbname: PostgreSQL database name
                - dbuser: PostgreSQL user
                - catalog_name: Name of the DuckLake catalog
                - data_path: Path to data storage (e.g., gs://bucket/path)
                - metadata_schema: PostgreSQL schema for metadata
            conn: Optional existing DuckDB connection to reuse.
        """
        self._owns_conn: bool = conn is None
        self._conn: duckdb.DuckDBPyConnection = conn or duckdb.connect()

        # Register GCS filesystem
        fs = gcsfs.GCSFileSystem()
        self._conn.register_filesystem(fs)

        # Load extensions
        for ext in ("ducklake", "postgres"):
            self._conn.sql(f"INSTALL {ext}")
            self._conn.sql(f"LOAD {ext}")

        # Attach catalog
        self._conn.sql(
            f"""
            ATTACH 'ducklake:postgres:
                dbname={db_config["dbname"]}
                user={db_config["dbuser"]}
                host=localhost
            ' AS {db_config["catalog_name"]}
            (DATA_PATH '{db_config["data_path"]}',
             METADATA_SCHEMA {db_config["metadata_schema"]});
            """
        )
        self._conn.sql(f"USE {db_config['catalog_name']}")

    def execute(self, sql: str, parameters: list[Any] | None = None) -> Any:
        """Execute SQL statement.

        Args:
            sql: SQL statement to execute.
            parameters: Optional list of parameters for parameterized queries.

        Returns:
            DuckDB relation with query results.
        """
        if parameters is not None:
            return self._conn.execute(sql, parameters)
        return self._conn.execute(sql)

    def sql(self, query: str) -> Any:
        """Execute SQL query.

        Args:
            query: SQL query to execute.

        Returns:
            DuckDB relation with query results.
        """
        return self._conn.sql(query)

    def register(self, name: str, obj: str) -> None:
        """Register Python object as virtual table.

        Args:
            name: Name for the virtual table.
            obj: Python object (typically a DataFrame) to register.
        """
        self._conn.register(name, obj)

    def close(self) -> None:
        """Close connection if owned by this instance."""
        if self._owns_conn:
            self._conn.close()

    @property
    def owns_connection(self) -> bool:
        """Whether this instance owns the connection."""
        return self._owns_conn
