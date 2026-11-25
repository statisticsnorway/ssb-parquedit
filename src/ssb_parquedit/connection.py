"""DuckDB connection management with DuckLake catalog support."""

import logging
from typing import Any

import duckdb
import duckdb_core_ext_ducklake
import gcsfs

logger = logging.getLogger(__name__)

_CLOSED_MSG = "Connection is closed."

def connect(extensions: tuple[str, ...] = ()) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection and load extensions installed from PyPI.

    Extensions come from the ``duckdb-core-ext-*`` packages, which place their
    binaries in ``site-packages/.duckdb_extensions``. The ``duckdb`` Python
    module does not search that directory by itself, so it is passed in via
    ``extension_directories``. Extensions are loaded explicitly, since
    autoloading only checks the first directory and would otherwise try to
    download them.

    Args:
        extensions: Extension names to load, e.g. ``("ducklake", "postgres")``.

    Returns:
        A DuckDB connection with the given extensions loaded.
    """
    conn = duckdb.connect(
        config={
            "extension_directories": [duckdb_core_ext_ducklake.get_extensions_dir()]
        }
    )
    for ext in extensions:
        conn.sql(f"LOAD {ext}")
    return conn


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

        Creates a new DuckDB connection, loads the required extensions,
        registers the GCS filesystem, and attaches the DuckLake
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
        self._conn = connect(("httpfs", "ducklake", "postgres"))

        fs = gcsfs.GCSFileSystem()

        self._conn.register_filesystem(fs)

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
