"""Fixtures for local integration tests - real DuckDB, no mocks."""

import shutil
import tempfile
from collections.abc import Generator

import pytest

from ssb_parquedit.local import LocalDuckDBConnection
from ssb_parquedit.parquedit import ParquEdit

# ============ Disable autouse-stubs from tests/conftest.py ============


@pytest.fixture(autouse=True)
def stub_external_modules() -> Generator[None]:
    """Override stub-fixture from tests/conftest.py - integration tests use real libs."""
    yield


# ============ Local test environment ============


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
