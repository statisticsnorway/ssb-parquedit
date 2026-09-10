"""Shared fixtures for unit tests."""

import shutil
import tempfile
from collections.abc import Generator

import pandas as pd
import polars as pl
import pytest

from ssb_parquedit.local import LocalDuckDBConnection


@pytest.fixture
def tmp_storage() -> Generator[str]:
    """Temporary directory that is removed after the test."""
    d = tempfile.mkdtemp(prefix="parquedit_unit_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def conn(tmp_storage: str) -> Generator[LocalDuckDBConnection]:
    """Live LocalDuckDBConnection, closed after the test."""
    c = LocalDuckDBConnection(data_path=tmp_storage)
    yield c
    c.close()


@pytest.fixture
def df_with_long_column_names() -> pd.DataFrame:
    """DataFrame mixing valid columns with ones over Postgres's 63-byte identifier limit."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Oslo", "Bergen"],
            "a" * 64: [1.0, 2.0],  # 64 ASCII chars = 64 bytes, over the limit
            # 64 chars but 67 UTF-8 bytes because of æ/ø/å (real-world trigger)
            "distriktstilskuddforfruktbærveksthusgrønnsakerinklsalatpåfriland": [
                1.0,
                2.0,
            ],
        }
    )


@pytest.fixture
def polars_df_with_long_column_names() -> pl.DataFrame:
    """Polars equivalent of df_with_long_column_names, over the 63-byte identifier limit."""
    return pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["Oslo", "Bergen"],
            "a" * 64: [1.0, 2.0],  # 64 ASCII chars = 64 bytes, over the limit
            # 64 chars but 67 UTF-8 bytes because of æ/ø/å (real-world trigger)
            "distriktstilskuddforfruktbærveksthusgrønnsakerinklsalatpåfriland": [
                1.0,
                2.0,
            ],
        }
    )
