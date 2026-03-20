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
) -> Generator[None]:
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
        class StringDtype:
            """Fake StringDtype for pandas mocking."""

            pass

        class DataFrame:
            def __init__(self) -> None:
                self.dtypes: dict[Any, Any] = {}
                self._data: dict[str, list[Any]] = {}

            @property
            def columns(self) -> list[str]:
                """Return column names."""
                return list(self._data.keys())

            def copy(self) -> "FakePandas.DataFrame":
                """Return a shallow copy of the DataFrame."""
                df = FakePandas.DataFrame()
                df._data = self._data.copy()
                df.dtypes = self.dtypes.copy()
                return df

            def __len__(self) -> int:
                """Return the number of rows in the DataFrame."""
                if self._data:
                    return len(next(iter(self._data.values())))
                return 0

            def insert(self, loc: int, column: str, value: Any) -> None:
                """Insert a column into the DataFrame at the given location."""
                self._data[column] = value if isinstance(value, list) else [value]

            def astype(self, dtype_dict: dict[str, Any]) -> "FakePandas.DataFrame":
                """Return a copy with specified columns converted to new types."""
                return self.copy()

            def __getitem__(self, col: str) -> "FakePandas.Series":
                """Return a Series for the given column."""
                return FakePandas.Series(self._data.get(col, []))

        class Series:
            def __init__(self, data: list[Any]) -> None:
                self.dtype = object
                self._data = data

            def astype(self, dtype: Any) -> "FakePandas.Series":
                return FakePandas.Series(self._data)

            def where(self, cond: Any, other: Any = None) -> "FakePandas.Series":
                return FakePandas.Series(self._data)

            def isna(self) -> "FakePandas.Series":
                return FakePandas.Series([False] * len(self._data))

            def notna(self) -> "FakePandas.Series":
                return FakePandas.Series([True] * len(self._data))

        @staticmethod
        def to_numeric(series: Any, errors: str = "raise") -> "FakePandas.Series":
            """Stub for pd.to_numeric."""
            return FakePandas.Series([])

    # Fake pyarrow module
    class FakePyArrow:
        class Table:
            def __init__(self) -> None:
                self.schema = FakePyArrow.Schema([])

            @staticmethod
            def from_pandas(
                df: Any, preserve_index: bool = True
            ) -> "FakePyArrow.Table":
                t = FakePyArrow.Table()
                t.schema = FakePyArrow.Schema(list(df.columns))
                return t

        class Schema:
            def __init__(self, names: list[str]) -> None:
                self.names = names

    monkeypatch.setitem(sys.modules, "duckdb", FakeDuckDB())
    monkeypatch.setitem(sys.modules, "gcsfs", FakeGCSFS())
    monkeypatch.setitem(sys.modules, "pandas", FakePandas())
    monkeypatch.setitem(sys.modules, "pyarrow", FakePyArrow())

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
    conn.sql = MagicMock()
    conn.register = MagicMock()
    conn.register_filesystem = MagicMock()
    conn.close = MagicMock()

    # DESCRIBE and other fetchall() calls return empty list by default
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    conn.execute.return_value = mock_result

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
