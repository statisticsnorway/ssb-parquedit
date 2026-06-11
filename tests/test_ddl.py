"""Unit tests for DDLOperations — mocks GCS and DuckLake to test logic branches."""

import logging
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from ssb_parquedit.ddl import DDLOperations
from ssb_parquedit.local import LocalDuckDBConnection
from ssb_parquedit.parquedit import ParquEdit

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_conn() -> MagicMock:
    conn = MagicMock()
    schema_result = MagicMock()
    schema_result.fetchone.return_value = ("main",)
    conn.execute.return_value = schema_result
    return conn


# ── _get_table_location ───────────────────────────────────────────────────────


class TestGetTableLocation:
    def test_returns_data_path_schema_table(self, mock_conn: MagicMock) -> None:
        ddl = DDLOperations(mock_conn, {"data_path": "gs://bucket/data"})
        assert ddl._get_table_location("my_table") == "gs://bucket/data/main/my_table"

    def test_raises_when_no_data_path(self, mock_conn: MagicMock) -> None:
        ddl = DDLOperations(mock_conn, {})
        with pytest.raises(RuntimeError, match="no data_path configured"):
            ddl._get_table_location("my_table")

    def test_raises_when_no_db_config(self, mock_conn: MagicMock) -> None:
        ddl = DDLOperations(mock_conn, None)
        with pytest.raises(RuntimeError):
            ddl._get_table_location("my_table")

    def test_falls_back_to_main_when_schema_query_raises(self) -> None:
        conn = MagicMock()
        conn.execute.side_effect = Exception("db error")
        ddl = DDLOperations(conn, {"data_path": "gs://bucket/data"})
        assert ddl._get_table_location("my_table") == "gs://bucket/data/main/my_table"

    def test_falls_back_to_main_when_schema_row_is_none(self) -> None:
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None
        ddl = DDLOperations(conn, {"data_path": "gs://bucket/data"})
        assert ddl._get_table_location("my_table") == "gs://bucket/data/main/my_table"


# ── _expire_snapshots ─────────────────────────────────────────────────────────


class TestExpireSnapshots:
    def test_logs_warning_when_current_database_returns_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None
        ddl = DDLOperations(conn, {})
        with caplog.at_level(logging.WARNING):
            ddl._expire_snapshots("my_table")
        assert "no active catalog" in caplog.text

    def test_logs_error_when_current_database_query_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = MagicMock()
        conn.execute.side_effect = Exception("connection error")
        ddl = DDLOperations(conn, {})
        with caplog.at_level(logging.ERROR):
            ddl._expire_snapshots("my_table")
        assert "Could not determine current catalog" in caplog.text

    def test_returns_early_when_no_snapshots(self) -> None:
        conn = MagicMock()
        current_db = MagicMock()
        current_db.fetchone.return_value = ("my_catalog",)
        snapshots = MagicMock()
        snapshots.fetchall.return_value = []
        conn.execute.side_effect = [current_db, snapshots]
        ddl = DDLOperations(conn, {})
        ddl._expire_snapshots("my_table")
        assert not any(
            "ducklake_expire_snapshots" in str(c) for c in conn.execute.call_args_list
        )

    def test_calls_ducklake_expire_with_snapshot_ids(self) -> None:
        conn = MagicMock()
        current_db = MagicMock()
        current_db.fetchone.return_value = ("my_catalog",)
        snapshots = MagicMock()
        snapshots.fetchall.return_value = [(1,), (2,), (3,)]
        conn.execute.side_effect = [current_db, snapshots, MagicMock()]
        ddl = DDLOperations(conn, {})
        ddl._expire_snapshots("my_table")
        expire_calls = [
            c
            for c in conn.execute.call_args_list
            if "ducklake_expire_snapshots" in str(c)
        ]
        assert len(expire_calls) == 1
        assert "my_catalog" in str(expire_calls[0])
        assert "[1, 2, 3]" in str(expire_calls[0])

    def test_logs_error_when_expire_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = MagicMock()
        current_db = MagicMock()
        current_db.fetchone.return_value = ("my_catalog",)
        conn.execute.side_effect = [current_db, Exception("DuckLake error")]
        ddl = DDLOperations(conn, {})
        with caplog.at_level(logging.ERROR):
            ddl._expire_snapshots("my_table")
        assert "Error during snapshot expiration" in caplog.text


# ── _cleanup_gcs_files ────────────────────────────────────────────────────────


class TestCleanupGcsFiles:
    def test_logs_error_for_non_gcs_path(
        self, mock_conn: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        ddl = DDLOperations(mock_conn, {})
        with caplog.at_level(logging.ERROR):
            ddl._cleanup_gcs_files("/local/path/table", "my_table")
        assert "Invalid GCS path format" in caplog.text

    def test_logs_warning_when_path_not_found_in_gcs(
        self, mock_conn: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        with patch("ssb_parquedit.ddl.gcsfs.GCSFileSystem") as mock_fs_cls:
            mock_fs = MagicMock()
            mock_fs.exists.return_value = False
            mock_fs_cls.return_value = mock_fs
            ddl = DDLOperations(mock_conn, {})
            with caplog.at_level(logging.WARNING):
                ddl._cleanup_gcs_files("gs://bucket/table", "my_table")
        assert "not found in GCS" in caplog.text
        mock_fs.rm.assert_not_called()

    def test_deletes_files_when_path_exists(self, mock_conn: MagicMock) -> None:
        with patch("ssb_parquedit.ddl.gcsfs.GCSFileSystem") as mock_fs_cls:
            mock_fs = MagicMock()
            mock_fs.exists.return_value = True
            mock_fs_cls.return_value = mock_fs
            ddl = DDLOperations(mock_conn, {})
            ddl._cleanup_gcs_files("gs://bucket/table", "my_table")
        mock_fs.rm.assert_called_once_with("gs://bucket/table", recursive=True)


# ── drop_table(purge=True) ────────────────────────────────────────────────────


class TestDropTablePurge:
    def test_purge_drops_table(
        self, conn: LocalDuckDBConnection, tmp_storage: str
    ) -> None:
        pe = ParquEdit.from_connection(
            conn,
            db_config={
                "catalog_name": "test_catalog",
                "metadata_schema": "main",
                "data_path": tmp_storage,
            },
        )
        df = pd.DataFrame({"id": [1, 2], "name": ["Oslo", "Bergen"]})
        pe.create_table("cities", source=df, product_name="p", user_defined_id=["id"])
        pe.drop_table("cities", purge=True)
        assert not pe.exists("cities")

    def test_purge_with_edits_drops_table(
        self, conn: LocalDuckDBConnection, tmp_storage: str
    ) -> None:
        pe = ParquEdit.from_connection(
            conn,
            db_config={
                "catalog_name": "test_catalog",
                "metadata_schema": "main",
                "data_path": tmp_storage,
            },
        )
        df = pd.DataFrame({"id": [1, 2], "name": ["Oslo", "Bergen"]})
        pe.create_table(
            "cities", source=df, product_name="p", user_defined_id=["id"], fill=True
        )
        rowid = int(pe.view("cities")["rowid"].iloc[0])
        pe.edit("cities", rowid, {"name": "Oslo edited"}, "OTHER", "test")
        pe.drop_table("cities", purge=True)
        assert not pe.exists("cities")

    def test_purge_location_failure_still_drops_table(
        self, mock_conn: MagicMock
    ) -> None:
        mock_conn.execute.side_effect = [
            Exception("location error"),  # _get_table_location -> CURRENT_SCHEMA raises
            MagicMock(),  # DROP TABLE
        ]
        ddl = DDLOperations(mock_conn, {})
        with pytest.raises(RuntimeError):
            ddl._get_table_location("cities")

        mock_conn.execute.side_effect = None
        mock_conn.execute.return_value = MagicMock()
        ddl.conn.execute("DROP TABLE cities")
