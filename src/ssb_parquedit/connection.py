"""DuckDB connection management with DuckLake catalog support."""

import logging
import re
from datetime import datetime
from types import TracebackType
from typing import Any

import duckdb
import gcsfs

from .functions import get_dapla_environment

# Configure module-level logger
logger = logging.getLogger(__name__)


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
        self._conn.sql(f"""
            ATTACH 'ducklake:postgres:
                dbname={db_config["dbname"]}
                user={db_config["dbuser"]}
                host=localhost
            ' AS {db_config["catalog_name"]}
            (DATA_PATH '{db_config["data_path"]}',
             METADATA_SCHEMA {db_config["metadata_schema"]},
             DATA_INLINING_ROW_LIMIT 300,
             AUTOMATIC_MIGRATION TRUE);
            """)
        self._conn.sql(f"USE {db_config['catalog_name']}")

    def __enter__(self) -> "DuckDBConnection":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Context manager exit, closes connection."""
        self.close()

    def execute(self, sql: str, parameters: list[Any] | None = None) -> Any:
        """Execute SQL statement with DROP operation enforcement.

        Enforces environment-based restrictions on DROP operations:
        - In TEST environment: DROP operations are allowed and logged
        - In other environments: DROP operations are blocked with PermissionError

        Args:
            sql: SQL statement to execute.
            parameters: Optional list of parameters for parameterized queries.

        Returns:
            DuckDB relation with query results.
        """
        # Check for DROP operations and enforce environment restrictions
        self._check_drop_operation(sql)

        if parameters is not None:
            return self._conn.execute(sql, parameters)
        return self._conn.execute(sql)

    def _check_drop_operation(self, sql: str) -> None:
        """Check for DROP operations and enforce/log them based on environment.

        Args:
            sql: SQL statement to check.

        Raises:
            PermissionError: If DROP operation is attempted in non-TEST environment.
        """
        # Pattern to match DROP TABLE, DROP VIEW, etc.
        drop_pattern = r"\bDROP\s+(TABLE|VIEW|DATABASE|SCHEMA)\b"
        if re.search(drop_pattern, sql, re.IGNORECASE):
            environment = get_dapla_environment()

            if environment != "test":
                raise PermissionError(
                    f"DROP operations are only allowed in TEST environment. "
                    f"Current environment: {environment or 'not set'}. "
                    f"Set DAPLA_ENVIRONMENT=test to enable DROP operations."
                )

            # Log the DROP operation in TEST environment
            table_match = re.search(
                r"\bDROP\s+(?:TABLE|VIEW|DATABASE|SCHEMA)\s+(\w+)",
                sql,
                re.IGNORECASE,
            )
            object_name = table_match.group(1) if table_match else "unknown"

            logger.warning(
                f"DROP operation executed in {environment.upper()} environment | "
                f"Object: {object_name} | Timestamp: {datetime.now().isoformat()}"
            )

    def sql(self, query: str) -> Any:
        """Execute SQL query with DROP operation enforcement.

        Enforces environment-based restrictions on DROP operations.

        Args:
            query: SQL query to execute.

        Returns:
            DuckDB relation with query results.
        """
        # Check for DROP operations and enforce environment restrictions
        self._check_drop_operation(query)

        return self._conn.sql(query)

    def register(self, name: str, obj: Any) -> None:
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
