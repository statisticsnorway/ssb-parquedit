"""Fixtures for local integration tests - real DuckDB, no mocks."""

import shutil
import tempfile
from collections.abc import Generator

import duckdb
import pytest

from ssb_parquedit.connection import DuckDBConnection
from ssb_parquedit.parquedit import ParquEdit

# ============ Disable autouse-stubs from tests/conftest.py ============


@pytest.fixture(autouse=True)
def stub_external_modules() -> Generator[None]:
    """Override stub-fixture from tests/conftest.py - integration tests use real libs."""
    yield


# ============ Local test environment ============


class LocalDuckDBConnection(DuckDBConnection):
    """Test version of DuckDBConnection.

    Bypasses __init__ entirely and uses:
    - ducklake:sqlite: instead of PostgreSQL
    - local tmpdir instead of GCS
    """

    def __init__(self, data_path: str) -> None:
        """Initialize local DuckDB connection with SQLite catalog.

        Bypasses the parent __init__ and connects to a local DuckLake instance
        backed by SQLite and local file storage instead of PostgreSQL and GCS.

        Args:
            data_path: Local directory used for both the SQLite catalog
                (catalog.db) and Parquet data files (data/).
        """
        self._conn = duckdb.connect()
        self._conn.sql("INSTALL sqlite; LOAD sqlite;")
        self._conn.sql("INSTALL ducklake; LOAD ducklake;")
        self._conn.sql(f"""
            ATTACH 'ducklake:sqlite:{data_path}/catalog.db' AS test_catalog
            (DATA_PATH '{data_path}/data',
             DATA_INLINING_ROW_LIMIT 300)
        """)
        self._conn.sql("USE test_catalog")


@pytest.fixture(autouse=True)
def dapla_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Set environment variables for all integration tests."""
    monkeypatch.setenv("DAPLA_ENVIRONMENT", "test")
    monkeypatch.setenv("DAPLA_GROUP_CONTEXT", "dapla-ffunk-developers")
    monkeypatch.setenv("DAPLA_USER", "test-user@ssb.no")
    yield


@pytest.fixture()
def tmp_storage() -> Generator[str]:
    d = tempfile.mkdtemp(prefix="parquedit_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def conn(tmp_storage: str) -> Generator[LocalDuckDBConnection]:
    c = LocalDuckDBConnection(data_path=tmp_storage)
    yield c
    c.close()


@pytest.fixture()
def pe(conn: LocalDuckDBConnection) -> ParquEdit:
    return ParquEdit.from_connection(
        conn,
        db_config={
            "catalog_name": "test_catalog",
            "metadata_schema": "main",  # SQLite uses 'main' as default schema
            "data_path": "",
        },
    )
