"""Shared test fixtures and configuration for ssb-parquedit tests."""

import importlib
import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---- Test scaffolding: stub external modules before importing the SUT ----
@pytest.fixture(autouse=True)
def stub_external_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Stub external heavy dependencies (duckdb, gcsfs, pandas) so tests run hermetically.

    We inject minimal fakes into sys.modules prior to importing ssb_parquedit.parquedit.
    """

    # Fake duckdb module with minimal API surface
    class FakeDuckDB:
        class DuckDBPyConnection:  # only for type hints; runtime uses MagicMock
            pass

        def connect(self) -> MagicMock:
            # Return a MagicMock connection to simulate owned connections
            conn = MagicMock()
            conn.sql = MagicMock()
            conn.execute = MagicMock()
            conn.register = MagicMock()
            conn.register_filesystem = MagicMock()
            conn.close = MagicMock()
            return conn

    # Fake gcsfs module with a GCSFileSystem type
    class FakeGCSFS:
        class GCSFileSystem:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.created = True

    # Fake pandas with just a DataFrame type for isinstance checks
    class FakePandas:
        class DataFrame:
            pass

    monkeypatch.setitem(sys.modules, "duckdb", FakeDuckDB())
    monkeypatch.setitem(sys.modules, "gcsfs", FakeGCSFS())
    monkeypatch.setitem(sys.modules, "pandas", FakePandas())

    yield

    # Cleanup is automatic by pytest monkeypatch fixture


@pytest.fixture
def sut() -> Any:
    """Import and return the ParquEdit class with stubs injected."""
    module = importlib.import_module("ssb_parquedit.parquedit")
    importlib.reload(module)
    return module.ParquEdit


@pytest.fixture
def fake_conn() -> MagicMock:
    """A MagicMock simulating a DuckDB connection."""
    conn = MagicMock()
    # Provide attributes/methods that ParquEdit expects
    # - register_filesystem(fs)
    # - sql(str)
    # - execute(str)
    # - register(name, obj)
    # - close()
    # Use wraps to capture SQL calls distinctly
    conn.sql = MagicMock()
    conn.execute = MagicMock()
    conn.register = MagicMock()
    conn.register_filesystem = MagicMock()
    conn.close = MagicMock()
    return conn


@pytest.fixture
def db_config() -> dict[str, str]:
    """Standard database configuration for tests."""
    return {
        "dbname": "testdb",
        "dbuser": "testuser",
        "catalog_name": "testcat",
        "data_path": "gs://bucket/path",
        "metadata_schema": "meta_schema",
    }


@pytest.fixture
def query_test_setup(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> tuple[Any, MagicMock]:
    """Setup fixture for query/view tests.

    Returns a tuple of (pe, fake_conn) with execute mocked and ready.
    """
    pe = sut(db_config=db_config, conn=fake_conn)

    # Mock the standard result from execute()
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result

    return pe, fake_conn
