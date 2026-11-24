"""DuckDB connection management with DuckLake catalog support."""

import logging
from pathlib import Path
from typing import Any

import duckdb
import gcsfs

logger = logging.getLogger(__name__)

_CLOSED_MSG = "Connection is closed."

_EXTENSION_DIR = Path(__file__).parent / "extentions"
_FILE_STEMS = {"postgres": "postgres_scanner", "sqlite": "sqlite_scanner"}


def install_extensions(conn: duckdb.DuckDBPyConnection, names: tuple[str, ...]) -> None:
    """Install and load DuckDB extensions, preferring bundled offline files.

    Files in the ``extentions`` folder are installed from disk (no network).
    Others fall back to the DuckDB repository. Extensions already installed
    (from the same bundled file) are only loaded, so the install happens once
    per DuckDB version. FORCE avoids "origin is different" errors when the
    extension was cached from another source.

    Args:
        conn: DuckDB connection to install the extensions into.
        names: Extension names, e.g. ``("ducklake", "postgres")``.
    """
    installed = {
        row[0]: (row[1], row[2])
        for row in conn.sql(
            "SELECT extension_name, installed_from, install_path "
            "FROM duckdb_extensions() WHERE installed"
        ).fetchall()
    }
    for name in names:
        stem = _FILE_STEMS.get(name, name)
        path = _EXTENSION_DIR / f"{stem}.duckdb_extension"
        if path.is_file():
            if not _is_installed_from(installed.get(stem), path):
                conn.sql(f"FORCE INSTALL '{path}'")
            conn.sql(f"LOAD {stem}")
        else:
            if stem not in installed:
                conn.sql(f"FORCE INSTALL {name}")
            conn.sql(f"LOAD {name}")


def _is_installed_from(entry: tuple[str, str] | None, path: Path) -> bool:
    """Check if the cached extension was installed from ``path`` and is unchanged."""
    if entry is None:
        return False
    installed_from, install_path = entry
    cached = Path(install_path)
    return (
        installed_from == str(path)
        and cached.is_file()
        and cached.stat().st_size == path.stat().st_size
    )


class DuckDBConnection:
    """Manages DuckDB connection with DuckLake catalog integration.

    Handles connection lifecycle, GCS filesystem registration, DuckLake
    and Postgres extension loading, and catalog attachment.

    Example:
        >>> # doctest: +SKIP
        >>> config = {
        ...     "dbname": "mydb",
        ...     "dbuser": "user",
        ...     "catalog_name": "my_catalog",
        ...     "data_path": "gs://bucket/path",
        ...     "metadata_schema": "ducklake",
        ... }
        >>> conn = DuckDBConnection(config)
        >>> conn.execute("SELECT 1")
        >>> conn.close()
    """

    _conn: duckdb.DuckDBPyConnection | None = None
    _catalog_name: str | None = None

    def __init__(self, db_config: dict[str, str]) -> None:
        """Initialize DuckDB connection with DuckLake catalog.

        Creates a new DuckDB connection, registers the GCS filesystem,
        installs and loads required extensions, and attaches the DuckLake
        catalog backed by PostgreSQL.

        Args:
            db_config: Database configuration dict with the following keys:

                - ``dbname``: PostgreSQL database name.
                - ``dbuser``: PostgreSQL user.
                - ``host``: PostgreSQL host.
                - ``port_number``: PostgreSQL port.
                - ``catalog_name``: Name of the DuckLake catalog to attach.
                - ``data_path``: GCS path for data storage (e.g. ``gs://bucket/path``).
                - ``metadata_schema``: PostgreSQL schema for DuckLake metadata.

        """
        self._conn = duckdb.connect()

        fs = gcsfs.GCSFileSystem()

        self._conn.register_filesystem(fs)

        install_extensions(self._conn, ("httpfs", "ducklake", "postgres"))

        self._conn.sql(f"""
            ATTACH 'ducklake:postgres:
                dbname={db_config["dbname"]}
                user={db_config["dbuser"]}
                host={db_config["host"]}
                port={db_config["port_number"]}
            ' AS {db_config["catalog_name"]}
            (DATA_PATH '{db_config["data_path"]}',
            METADATA_SCHEMA {db_config["metadata_schema"]},
            DATA_INLINING_ROW_LIMIT 300,
            AUTOMATIC_MIGRATION TRUE);
            """)
        self._conn.sql(f"USE {db_config['catalog_name']}")

        self._catalog_name = db_config["catalog_name"]
        logger.info(
            "Connected to DuckLake catalog '%s' (data_path=%s)",
            db_config["catalog_name"],
            db_config["data_path"],
        )
    def execute(self, sql: str, parameters: list[Any] | None = None) -> Any:
        """Execute a SQL statement.

        Args:
            sql: SQL statement to execute.
            parameters: Optional list of parameters for parameterized queries.

        Returns:
            DuckDB relation containing the query results.

        Raises:
            RuntimeError: If the connection has been closed.

        Example:
            >>> # doctest: +SKIP
            >>> conn.execute("SELECT count(*) FROM my_table")
            >>> conn.execute("SELECT * FROM my_table WHERE id = ?", [42])
        """
        if self._conn is None:
            logger.error(_CLOSED_MSG)
            raise RuntimeError(_CLOSED_MSG)
        if parameters is not None:
            return self._conn.execute(sql, parameters)
        return self._conn.execute(sql)

    def sql(self, query: str) -> Any:
        """Execute a SQL query.

        Equivalent to ``execute`` but uses DuckDB's ``sql`` method, which
        accepts a broader range of statement types including multi-statement
        strings and DuckDB-specific syntax.

        Args:
            query: SQL query to execute.

        Returns:
            DuckDB relation containing the query results.

        Raises:
            RuntimeError: If the connection has been closed.

        Example:
            >>> # doctest: +SKIP
            >>> conn.sql("CALL ducklake_flush_inlined_data('my_catalog')")
        """
        if self._conn is None:
            logger.error(_CLOSED_MSG)
            raise RuntimeError(_CLOSED_MSG)
        return self._conn.sql(query)

    def register(self, name: str, obj: Any) -> None:
        """Register a Python object as a virtual table in DuckDB.

        Makes a Python object (typically a DataFrame) queryable as a SQL
        table within the current connection.

        Args:
            name: Name to assign to the virtual table.
            obj: Python object to register, typically a ``pd.DataFrame``
                or ``pyarrow.Table``.

        Raises:
            RuntimeError: If the connection has been closed.

        Example:
            >>> # doctest: +SKIP
            >>> conn.register("staging", df)
            >>> conn.execute("INSERT INTO my_table SELECT * FROM staging")
        """
        if self._conn is None:
            logger.error(_CLOSED_MSG)
            raise RuntimeError(_CLOSED_MSG)
        self._conn.register(name, obj)

    def close(self) -> None:
        """Close the underlying DuckDB connection.

        After calling this method, any further calls to ``execute``, ``sql``,
        ``register``, or ``raw`` will raise a ``RuntimeError``.

        Example:
            >>> # doctest: +SKIP
            >>> conn.close()
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info(
                "Closed connection to DuckLake catalog '%s'", self._catalog_name
            )

    @property
    def raw(self) -> duckdb.DuckDBPyConnection:
        """The underlying DuckDB connection instance.

        Provides direct access to the raw ``duckdb.DuckDBPyConnection`` for
        use with external libraries such as Ibis that require a native DuckDB
        connection object.

        Returns:
            The underlying ``duckdb.DuckDBPyConnection`` instance.

        Raises:
            RuntimeError: If the connection has been closed.

        Example:
            >>> # doctest: +SKIP
            >>> import ibis
            >>> ibis_conn = ibis.duckdb.connect(conn=ducklake_conn.raw)
        """
        if self._conn is None:
            logger.error(_CLOSED_MSG)
            raise RuntimeError(_CLOSED_MSG)
        return self._conn
