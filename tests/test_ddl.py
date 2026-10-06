"""Unit tests for DDLOperations — mocks GCS and DuckLake to test logic branches."""

import logging
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
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

    def test_exception_raised(self) -> None:
        mock_conn = MagicMock()
        db_config = MagicMock()
        ddl = DDLOperations(mock_conn, db_config)

        table_location = "gs://test/example"
        table_name = "t1"

        with (
            patch("ssb_parquedit.ddl.gcsfs.GCSFileSystem") as mock_gcsf,
            patch("ssb_parquedit.ddl.logger") as mock_log,
        ):
            mock_gcsf.return_value.exists = MagicMock().side_effect
            mock_gcsf.exists.side_effect = Exception("failed")
            e = MagicMock()
            mock_log.error = e
            ddl._cleanup_gcs_files(table_location, table_name)

        e.assert_called_once_with(
            f"Failed to clean up GCS files for {table_name} at {table_location}: 'NoneType' object is not callable. Files may need manual cleanup. Verify path and GCS permissions."
        )


# ── _cleanup_local_files ────────────────────────────────────────────────────────


class TestCleanupLocalFiles:
    def test_path_does_not_exists(self) -> None:
        mock_conn = MagicMock()
        db_config = MagicMock()
        ddl = DDLOperations(mock_conn, db_config)
        table_location = "/test/example"
        table_name = "t1"
        with (
            patch("ssb_parquedit.ddl.Path") as mock_path,
            patch("ssb_parquedit.ddl.logger") as mock_log,
        ):
            warn = MagicMock()
            mock_log.warning = warn
            mock_path.return_value.exists.return_value = False
            ddl._cleanup_local_files(table_location, table_name)

        warn.assert_called_once_with(
            f"Table location not found locally: {table_location}. "
            f"Data may have already been deleted or path is incorrect."
        )

    def test_failes_to_remove(self) -> None:
        mock_conn = MagicMock()
        db_config = MagicMock()
        ddl = DDLOperations(mock_conn, db_config)
        table_location = "/test/test"
        table_name = "t1"
        with (
            patch("ssb_parquedit.ddl.shutil") as mock_shu,
            patch("ssb_parquedit.ddl.logger") as mock_log,
            patch("ssb_parquedit.ddl.Path") as mock_path,
        ):
            mock_path.return_value.exists.return_value = True
            rm = MagicMock()
            rm.side_effect = Exception("test")
            mock_shu.rmtree = rm
            ex = MagicMock()
            mock_log.exception = ex
            ddl._cleanup_local_files(table_location, table_name)

        rm.assert_called_once()
        ex.assert_called_once_with(
            f"Failed to clean up local files for {table_name} at {table_location}"
            f"Files may need manual cleanup."
        )


# ── drop_table(cleanup=True) ────────────────────────────────────────────────────


class TestDropTable:
    def test_cleanup_drops_table(
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
        pe.drop_table("cities", cleanup=True)
        assert not pe.exists("cities")

    def test_cleanup_edits_drops_table(
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
        pe.drop_table("cities", cleanup=True)
        assert not pe.exists("cities")

    def test_cleanup_location_failure_still_drops_table(
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

    def test_invalid_table_name(self):
        mock_conn = MagicMock()
        ddl = DDLOperations(mock_conn)
        with pytest.raises(ValueError):
            ddl.drop_table("BAD NAME")

    def test_cleanup_table_location_not_found(self):
        mock_conn = MagicMock()
        mock_conn.db_config = None
        ddl = DDLOperations(mock_conn)
        with patch("ssb_parquedit.ddl.logger") as log:
            w = MagicMock()
            log.warning = w
            ddl.drop_table("t1", cleanup=True)

        assert (
            call(
                "Could not retrieve table location for t1: Cannot determine table location for t1: no data_path configured.. Proceeding with drop only, GCS files may need manual cleanup."
            )
            in w.call_args_list
        )

    def test_cleanup_remote_conn(self):
        mock_conn = MagicMock()
        ddl = DDLOperations(mock_conn)
        ddl._cleanup_gcs_files = MagicMock()
        ddl._get_table_location = MagicMock()
        table_location = "test"
        table_name = "t1"
        ddl._get_table_location.return_value = table_location
        with patch("ssb_parquedit.ddl.gcsfs.GCSFileSystem"):
            ddl.drop_table(table_name, True)

        ddl._cleanup_gcs_files.assert_called_once_with(table_location, table_name)


# ── create_table: column name length validation ─────────────────────────────


class TestCreateTable:
    """create_table() must reject column names over Postgres's 63-byte identifier limit."""

    LONG_ASCII_NAME = "a" * 64
    # 64 chars but 67 UTF-8 bytes because of æ/ø/å — the real-world trigger.
    LONG_MULTIBYTE_NAME = (
        "distriktstilskuddforfruktbærveksthusgrønnsakerinklsalatpåfriland"
    )

    def test_raises_for_long_ascii_column_from_dataframe(
        self, conn: LocalDuckDBConnection, df_with_long_column_names: pd.DataFrame
    ) -> None:
        ddl = DDLOperations(conn)
        with pytest.raises(ValueError, match="63-byte"):
            ddl.create_table("t1", df_with_long_column_names)

    def test_raises_for_long_multibyte_column_from_dataframe(
        self, conn: LocalDuckDBConnection
    ) -> None:
        df = pd.DataFrame({"id": [1], self.LONG_MULTIBYTE_NAME: [1.0]})
        ddl = DDLOperations(conn)
        with pytest.raises(ValueError, match="63-byte"):
            ddl.create_table("t1", df)

    def test_accepts_column_name_at_63_bytes(self, conn: LocalDuckDBConnection) -> None:
        df = pd.DataFrame({"id": [1], "a" * 63: [1.0]})
        DDLOperations(conn).create_table("t1", df)  # must not raise

    def test_raises_for_long_ascii_column_from_polars_dataframe(
        self,
        conn: LocalDuckDBConnection,
        polars_df_with_long_column_names: pl.DataFrame,
    ) -> None:
        ddl = DDLOperations(conn)
        with pytest.raises(ValueError, match="63-byte"):
            ddl.create_table("t1", polars_df_with_long_column_names)

    def test_raises_for_long_multibyte_column_from_polars_dataframe(
        self, conn: LocalDuckDBConnection
    ) -> None:
        df = pl.DataFrame({"id": [1], self.LONG_MULTIBYTE_NAME: [1.0]})
        ddl = DDLOperations(conn)
        with pytest.raises(ValueError, match="63-byte"):
            ddl.create_table("t1", df)

    def test_accepts_polars_column_name_at_63_bytes(
        self, conn: LocalDuckDBConnection
    ) -> None:
        df = pl.DataFrame({"id": [1], "a" * 63: [1.0]})
        DDLOperations(conn).create_table("t1", df)  # must not raise

    def test_raises_for_long_property_name_from_schema(
        self, conn: LocalDuckDBConnection
    ) -> None:
        schema = {
            "properties": {
                "id": {"type": "integer"},
                self.LONG_ASCII_NAME: {"type": "string"},
            }
        }
        ddl = DDLOperations(conn)
        with pytest.raises(ValueError, match="63-byte"):
            ddl.create_table("t1", schema)

    def test_raises_for_long_column_name_from_parquet(
        self,
        conn: LocalDuckDBConnection,
        tmp_storage: str,
        df_with_long_column_names: pd.DataFrame,
    ) -> None:
        parquet_path = str(Path(tmp_storage) / "wide.parquet")
        table = pa.Table.from_pandas(df_with_long_column_names, preserve_index=False)
        pq.write_table(table, parquet_path)

        ddl = DDLOperations(conn)
        with pytest.raises(ValueError, match="63-byte"):
            ddl.create_table("t1", parquet_path)

    def test_invalid_table_name(self):
        mock_conn = MagicMock()
        ddl = DDLOperations(mock_conn)
        with pytest.raises(ValueError):
            ddl.create_table("BAD NAME", "test")

    def test_invalid_source(self):
        mock_conn = MagicMock()
        ddl = DDLOperations(mock_conn)
        with pytest.raises(
            TypeError,
            match="source must be a DataFrame, JSON Schema dict, or gs:// Parquet path",
        ):
            ddl.create_table("t1", 1)

    def test_part_columns_greater_than_zero(self):
        mock_conn = MagicMock()
        part_columns = ["col_a", "col_b"]
        schema = {
            "properties": {
                "id": {"type": "integer"},
                "a": {"type": "string"},
            }
        }
        ddl = DDLOperations(mock_conn)
        table_name = "t1"
        cols = ", ".join(part_columns)
        ddl.create_table(table_name, schema, part_columns=part_columns)
        assert (
            call(f"ALTER TABLE {table_name} SET PARTITIONED BY ({cols})")
            in mock_conn.execute.call_args_list
        )
