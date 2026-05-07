"""Fixtures for local integration tests – ekte DuckDB, ingen mocks."""

import shutil
import tempfile
from collections.abc import Generator

import duckdb
import pytest

from ssb_parquedit.connection import DuckDBConnection
from ssb_parquedit.parquedit import ParquEdit


# ============ Deaktiver autouse-stubs fra tests/conftest.py ============

@pytest.fixture(autouse=True)
def stub_external_modules() -> Generator[None, None, None]:
    """Overstyr stub-fixture fra tests/conftest.py – integrasjonstester bruker ekte libs."""
    yield


# ============ Lokalt testmiljø ============

class LocalDuckDBConnection(DuckDBConnection):
    """Testversjon av DuckDBConnection.

    Omgår __init__ fullstendig og bruker:
    - ducklake:sqlite: i stedet for PostgreSQL
    - lokal tmpdir i stedet for GCS
    """

    def __init__(self, data_path: str) -> None:
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
def dapla_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Setter miljøvariabler for alle integrasjonstester."""
    monkeypatch.setenv("DAPLA_ENVIRONMENT", "test")
    monkeypatch.setenv("DAPLA_GROUP_CONTEXT", "dapla-ffunk-developers")
    monkeypatch.setenv("DAPLA_USER", "test-user@ssb.no")
    yield


@pytest.fixture()
def tmp_storage() -> Generator[str, None, None]:
    d = tempfile.mkdtemp(prefix="parquedit_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def conn(tmp_storage: str) -> Generator[LocalDuckDBConnection, None, None]:
    c = LocalDuckDBConnection(data_path=tmp_storage)
    yield c
    c.close()


@pytest.fixture()
def pe(conn: LocalDuckDBConnection) -> ParquEdit:
    return ParquEdit.from_connection(
        conn,
        db_config={
            "catalog_name": "test_catalog",
            "metadata_schema": "main",  # SQLite bruker 'main' som default schema
            "data_path": "",
        },
    )