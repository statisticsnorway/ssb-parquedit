"""Shared fixtures for unit tests."""

import shutil
import tempfile
from collections.abc import Generator

import duckdb
import pytest

from ssb_parquedit.connection import DuckDBConnection


class LocalDuckDBConnection(DuckDBConnection):
    """A real DuckDBConnection backed by DuckLake/SQLite for unit testing.

    Bypasses DuckDBConnection.__init__ to avoid GCS and PostgreSQL dependencies,
    using a local SQLite catalog and a temporary directory for data files instead.
    """

    def __init__(self, data_path: str) -> None:
        """Create a DuckLake connection backed by SQLite at the given data_path."""
        self._conn = duckdb.connect()
        self._conn.sql("INSTALL sqlite; LOAD sqlite;")
        self._conn.sql("INSTALL ducklake; LOAD ducklake;")
        self._conn.sql(f"""
            ATTACH 'ducklake:sqlite:{data_path}/catalog.db' AS test_catalog
            (DATA_PATH '{data_path}/data',
             DATA_INLINING_ROW_LIMIT 300)
        """)
        self._conn.sql("USE test_catalog")


@pytest.fixture()
def tmp_storage() -> Generator[str]:
    """Temporary directory that is removed after the test."""
    d = tempfile.mkdtemp(prefix="parquedit_unit_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def conn(tmp_storage: str) -> Generator[LocalDuckDBConnection]:
    """Live LocalDuckDBConnection, closed after the test."""
    c = LocalDuckDBConnection(data_path=tmp_storage)
    yield c
    c.close()
