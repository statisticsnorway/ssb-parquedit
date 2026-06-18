from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from ssb_parquedit.maintenance import MaintenanceOperations

LOGGER_NAME = "ssb_parquedit.maintenance"

# (connection, catalog_name, data_path)
DuckLake = tuple[duckdb.DuckDBPyConnection, str, Path]


@pytest.fixture
def ducklake(tmp_path: Path) -> Iterator[DuckLake]:
    """A local DuckLake attached with inlining disabled to force data files."""
    catalog = "test_lake"
    metadata = tmp_path / "metadata.ducklake"
    data_path = tmp_path / "data"
    data_path.mkdir()

    conn = duckdb.connect()
    conn.execute("INSTALL ducklake; LOAD ducklake;")
    conn.execute(
        f"ATTACH 'ducklake:{metadata}' AS {catalog} "
        f"(DATA_PATH '{data_path}/', DATA_INLINING_ROW_LIMIT 0)"
    )
    conn.execute(f"USE {catalog}")

    yield conn, catalog, data_path

    conn.close()


def _parquet_files(data_path: Path) -> list[Path]:
    return list(data_path.rglob("*.parquet"))


def test_merge_adjacent_compacts_files_and_preserves_rows(ducklake: DuckLake) -> None:
    conn, catalog, data_path = ducklake
    conn.execute("CREATE TABLE t (a INTEGER)")
    for i in range(8):
        conn.execute(f"INSERT INTO t VALUES ({i})")

    before_merge = _parquet_files(data_path)
    assert len(before_merge) == 8

    ops = MaintenanceOperations(conn, {"catalog_name": catalog})
    ops.merge_adjacent_files("t")

    after_merge = _parquet_files(data_path)
    assert len(after_merge) == 9  # 8 original files + 1 merged output file

    result = conn.execute("SELECT count(*) FROM t").fetchone()
    assert result is not None
    assert result[0] == 8


def test_second_merge_is_noop(ducklake: DuckLake, caplog: pytest.LogCaptureFixture) -> None:
    conn, catalog, _ = ducklake
    conn.execute("CREATE TABLE t (a INTEGER)")
    for i in range(4):
        conn.execute(f"INSERT INTO t VALUES ({i})")

    ops = MaintenanceOperations(conn, {"catalog_name": catalog})
    ops.merge_adjacent_files("t")

    with caplog.at_level("INFO", logger=LOGGER_NAME):
        ops.merge_adjacent_files("t")

    assert "No files to merge for table 't'." in caplog.text


def test_merge_logs_file_counts(
    ducklake: DuckLake, caplog: pytest.LogCaptureFixture
) -> None:
    conn, catalog, _ = ducklake
    conn.execute("CREATE TABLE t (a INTEGER)")
    for i in range(5):
        conn.execute(f"INSERT INTO t VALUES ({i})")

    ops = MaintenanceOperations(conn, {"catalog_name": catalog})
    with caplog.at_level("INFO", logger=LOGGER_NAME):
        ops.merge_adjacent_files("t")

    assert "Merged 5 files into 1 for table 't'." in caplog.text


def test_merge_requires_db_config(ducklake: DuckLake) -> None:
    conn, _, _ = ducklake
    ops = MaintenanceOperations(conn, None)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        ops.merge_adjacent_files("t")
