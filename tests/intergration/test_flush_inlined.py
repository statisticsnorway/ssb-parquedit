from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

# adjust to your actual module path
from ssb_parquedit.maintenance import MaintenanceOperations

LOGGER_NAME = "ssb_parquedit.maintenance"

# (connection, catalog_name, data_path)
DuckLake = tuple[duckdb.DuckDBPyConnection, str, Path]


@pytest.fixture
def ducklake(tmp_path: Path) -> Iterator[DuckLake]:
    """A local DuckLake attached with a high inlining threshold."""
    catalog = "test_lake"
    metadata = tmp_path / "metadata.ducklake"
    data_path = tmp_path / "data"
    data_path.mkdir()

    conn = duckdb.connect()
    conn.execute("INSTALL ducklake; LOAD ducklake;")
    conn.execute(
        f"ATTACH 'ducklake:{metadata}' AS {catalog} "
        f"(DATA_PATH '{data_path}/', DATA_INLINING_ROW_LIMIT 100)"
    )
    conn.execute(f"USE {catalog}")

    yield conn, catalog, data_path

    conn.close()


def _parquet_files(data_path: Path) -> list[Path]:
    return list(data_path.rglob("*.parquet"))


def test_flush_materializes_inlined_data(ducklake: DuckLake) -> None:
    conn, catalog, data_path = ducklake
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2), (3)")  # < limit -> inlined
    assert _parquet_files(data_path) == []  # inlined, nothing on disk

    ops = MaintenanceOperations(conn, {"catalog_name": catalog})
    ops.flush_inlined_table("t")

    assert _parquet_files(data_path)  # Parquet now written

    result = conn.execute("SELECT count(*) FROM t").fetchone()
    assert result is not None
    assert result[0] == 3  # data intact


def test_flush_logs_row_count(
    ducklake: DuckLake, caplog: pytest.LogCaptureFixture
) -> None:
    conn, catalog, _ = ducklake
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2), (3)")

    ops = MaintenanceOperations(conn, {"catalog_name": catalog})
    with caplog.at_level("INFO", logger=LOGGER_NAME):
        ops.flush_inlined_table("t")

    assert "Flushed 3 rows" in caplog.text


def test_second_flush_is_noop(
    ducklake: DuckLake, caplog: pytest.LogCaptureFixture
) -> None:
    conn, catalog, data_path = ducklake
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2)")
    ops = MaintenanceOperations(conn, {"catalog_name": catalog})

    ops.flush_inlined_table("t")
    before = set(_parquet_files(data_path))

    with caplog.at_level("INFO", logger=LOGGER_NAME):
        ops.flush_inlined_table("t")

    assert "No inlined data to flush" in caplog.text
    assert set(_parquet_files(data_path)) == before  # no new files


def test_flush_requires_db_config(ducklake: DuckLake) -> None:
    conn, _, _ = ducklake
    ops = MaintenanceOperations(conn, None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        ops.flush_inlined_table("t")
