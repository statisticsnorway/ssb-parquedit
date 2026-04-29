"""Shared test fixtures and configuration for ssb-parquedit tests."""

import importlib
import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---- Module-level fakes (defined once, reused across all fixtures) ----


class _FakeDataFrame:
    def __init__(self) -> None:
        self.dtypes: dict[Any, Any] = {}
        self._data: dict[str, list[Any]] = {}

    @property
    def columns(self) -> list[str]:
        """Return column names."""
        return list(self._data.keys())

    def copy(self) -> "_FakeDataFrame":
        """Return a shallow copy of the DataFrame."""
        df = _FakeDataFrame()
        df._data = self._data.copy()
        df.dtypes = self.dtypes.copy()
        return df

    def __setitem__(self, col: str, value: Any) -> None:
        """Set a column value."""
        self._data[col] = value

    def __len__(self) -> int:
        """Return the number of rows in the DataFrame."""
        if self._data:
            return len(next(iter(self._data.values())))
        return 0

    def insert(self, loc: int, column: str, value: Any) -> None:
        """Insert a column into the DataFrame at the given location."""
        self._data[column] = value if isinstance(value, list) else [value]

    def astype(self, dtype_dict: dict[str, Any]) -> "_FakeDataFrame":
        """Return a copy with specified columns converted to new types."""
        return self.copy()

    def __getitem__(self, col: str) -> "_FakeSeries":
        """Return a Series for the given column."""
        return _FakeSeries(self._data.get(col, []))


class _FakeSeries:
    def __init__(self, data: Any, dtype: Any = None) -> None:
        self.dtype = dtype or object
        self._data = data

    def astype(self, dtype: Any) -> "_FakeSeries":
        return _FakeSeries(self._data)

    def where(self, cond: Any, other: Any = None) -> "_FakeSeries":
        return _FakeSeries(self._data)

    def isna(self) -> "_FakeSeries":
        return _FakeSeries([False] * len(self._data))

    def notna(self) -> "_FakeSeries":
        return _FakeSeries([True] * len(self._data))


class _FakePandas:
    DataFrame = _FakeDataFrame
    Series = _FakeSeries

    class StringDtype:
        """Fake StringDtype for pandas mocking."""

        pass

    @staticmethod
    def to_numeric(series: Any, errors: str = "raise") -> "_FakeSeries":
        """Stub for pd.to_numeric."""
        return _FakeSeries([])


class _FakeDuckDB:
    class DuckDBPyConnection:  # only for type hints; runtime uses MagicMock
        pass

    def connect(self) -> MagicMock:
        conn = MagicMock()
        conn.sql = MagicMock()
        conn.execute = MagicMock()
        conn.register = MagicMock()
        conn.register_filesystem = MagicMock()
        conn.close = MagicMock()
        return conn


class _FakeGCSFS:
    class GCSFileSystem:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.created = True


class _FakePyArrowSchema:
    def __init__(self, names: list[str]) -> None:
        self.names = names


class _FakePyArrowTable:
    def __init__(self) -> None:
        self.schema = _FakePyArrowSchema([])

    @staticmethod
    def from_pandas(df: Any, preserve_index: bool = True) -> "_FakePyArrowTable":
        t = _FakePyArrowTable()
        t.schema = _FakePyArrowSchema(list(df.columns))
        return t


class _FakePyArrow:
    Table = _FakePyArrowTable
    Schema = _FakePyArrowSchema


# ---- Test scaffolding: stub external modules before importing the SUT ----


@pytest.fixture(autouse=True)
def stub_external_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None]:
    """Stub external heavy dependencies (duckdb, gcsfs, pandas) so tests run hermetically.

    We inject module-level fakes into sys.modules prior to importing ssb_parquedit.parquedit.
    Using module-level classes ensures the same class object is reused across fixtures,
    so isinstance checks pass correctly.
    """
    monkeypatch.setitem(sys.modules, "duckdb", _FakeDuckDB())
    monkeypatch.setitem(sys.modules, "gcsfs", _FakeGCSFS())
    monkeypatch.setitem(sys.modules, "pandas", _FakePandas)  # class, not instance
    monkeypatch.setitem(sys.modules, "pyarrow", _FakePyArrow())

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
