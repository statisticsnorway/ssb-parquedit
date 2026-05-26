"""Shared fixtures for unit tests."""

import shutil
import tempfile
from collections.abc import Generator

import pytest

from ssb_parquedit.local import LocalDuckDBConnection


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
