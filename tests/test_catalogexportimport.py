"""Unit tests for CatalogExportImport — mocks DuckDB/Postgres/GCS to test logic branches."""

from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pandas as pd
import pytest

from ssb_parquedit.catalogexportimport import CatalogExportImport
from ssb_parquedit.maintenance import MaintenanceOperations

DB_CONFIG = {
    "dbname": "metadata",
    "dbuser": "postgres",
    "data_path": "gs://bucket/data",
    "catalog_name": "test_catalog",
    "metadata_schema": "my_schema",
    "port_number": "5432",
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_conn() -> MagicMock:
    """A MagicMock connection where list_tables() resolves to no tables.

    Keeps export_catalog's flush/merge maintenance loop a no-op so tests can
    focus on the export/import orchestration itself.
    """
    conn = MagicMock()
    conn.execute.return_value.df.return_value = pd.DataFrame({"table_name": []})
    return conn


def _sql_calls(mock_conn: MagicMock) -> list[str]:
    """Flatten conn.sql(...) call args into a list of the SQL strings passed."""
    return [c.args[0] for c in mock_conn.sql.call_args_list]


# ── export_catalog: validation ────────────────────────────────────────────────


class TestExportCatalogValidation:
    def test_raises_when_db_config_none(self, mock_conn: MagicMock) -> None:
        export = CatalogExportImport(mock_conn, None)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="db_config is not initialized"):
            export.export_catalog()


# ── export_catalog: happy path ────────────────────────────────────────────────


class TestExportCatalogHappyPath:
    def test_uploads_backup_and_returns_expected_path(
        self, mock_conn: MagicMock
    ) -> None:
        mock_conn.sql.return_value.fetchall.return_value = []
        export = CatalogExportImport(mock_conn, DB_CONFIG)

        with patch("ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem") as fs_cls:
            fs = MagicMock()
            fs_cls.return_value = fs

            result = export.export_catalog(export_path="gs://bucket/backups")

        assert result.startswith("gs://bucket/data/catalog-export/")
        assert result.endswith("_my_schema.duckdb")
        fs.put.assert_called_once()
        local_path, remote_path = fs.put.call_args.args
        assert remote_path == "gs://bucket/backups/" + local_path.split("/")[-1]

    def test_uses_data_path_as_default_export_path(self, mock_conn: MagicMock) -> None:
        mock_conn.sql.return_value.fetchall.return_value = []
        export = CatalogExportImport(mock_conn, DB_CONFIG)

        with patch("ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem") as fs_cls:
            fs = MagicMock()
            fs_cls.return_value = fs

            export.export_catalog()

        _local_path, remote_path = fs.put.call_args.args
        assert remote_path.startswith("gs://bucket/data/catalog-export/")

    def test_runs_transaction_in_expected_order(self, mock_conn: MagicMock) -> None:
        mock_conn.sql.return_value.fetchall.return_value = []
        export = CatalogExportImport(mock_conn, DB_CONFIG)

        with patch("ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem"):
            export.export_catalog()

        calls = _sql_calls(mock_conn)
        assert calls[0] == "BEGIN"
        assert "ATTACH 'postgres:" in calls[1]
        assert "ATTACH 'duckdb:" in calls[2]
        assert "CREATE SCHEMA IF NOT EXISTS backup.my_schema" in calls[3]
        assert "COMMIT" in calls
        assert "DETACH catalog_db;" in calls
        assert "DETACH backup;" in calls

    def test_copies_every_table_from_the_catalog_schema(
        self, mock_conn: MagicMock
    ) -> None:
        mock_conn.sql.return_value.fetchall.return_value = [
            ("table_a",),
            ("table_b",),
        ]
        export = CatalogExportImport(mock_conn, DB_CONFIG)

        with patch("ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem"):
            export.export_catalog()

        calls = _sql_calls(mock_conn)
        assert any(
            "CREATE OR REPLACE TABLE backup.my_schema.table_a" in c for c in calls
        )
        assert any(
            "CREATE OR REPLACE TABLE backup.my_schema.table_b" in c for c in calls
        )

    def test_flushes_and_merges_every_table_before_export(
        self, mock_conn: MagicMock
    ) -> None:
        mock_conn.execute.return_value.df.return_value = pd.DataFrame(
            {"table_name": ["cities"]}
        )
        mock_conn.sql.return_value.fetchall.return_value = []
        export = CatalogExportImport(mock_conn, DB_CONFIG)

        with (
            patch("ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem"),
            patch.object(MaintenanceOperations, "flush_inlined_table") as flush_mock,
            patch.object(MaintenanceOperations, "merge_adjacent_files") as merge_mock,
        ):
            export.export_catalog()

        flush_mock.assert_called_once_with("cities")
        merge_mock.assert_called_once_with("cities")


# ── export_catalog: failure handling ─────────────────────────────────────────


class TestExportCatalogFailureHandling:
    def test_rolls_back_and_reraises_on_failure(self, mock_conn: MagicMock) -> None:
        def sql_side_effect(query: str, *args: object) -> MagicMock:
            if "CREATE SCHEMA" in query:
                raise RuntimeError("boom")
            return MagicMock()

        mock_conn.sql.side_effect = sql_side_effect
        export = CatalogExportImport(mock_conn, DB_CONFIG)

        with pytest.raises(RuntimeError, match="boom"):
            export.export_catalog()

        assert call("ROLLBACK") in mock_conn.sql.call_args_list

    def test_does_not_upload_to_gcs_on_failure(self, mock_conn: MagicMock) -> None:
        def sql_side_effect(query: str, *args: object) -> MagicMock:
            if "CREATE SCHEMA" in query:
                raise RuntimeError("boom")
            return MagicMock()

        mock_conn.sql.side_effect = sql_side_effect
        export = CatalogExportImport(mock_conn, DB_CONFIG)

        with patch("ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem") as fs_cls:
            fs = MagicMock()
            fs_cls.return_value = fs
            with pytest.raises(RuntimeError):
                export.export_catalog()

        fs.put.assert_not_called()


# ── import_catalog: validation ────────────────────────────────────────────────


class TestImportCatalogValidation:
    def test_raises_when_db_config_none(self, mock_conn: MagicMock) -> None:
        import_ = CatalogExportImport(mock_conn, None)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="db_config is not initialized"):
            import_.import_catalog("gs://bucket/backups/backup.duckdb")

    def test_returns_none_without_raising(self, mock_conn: MagicMock) -> None:
        mock_conn.sql.return_value.fetchall.return_value = []
        import_ = CatalogExportImport(mock_conn, DB_CONFIG)
        import_.import_catalog("gs://bucket/backups/backup.duckdb")  # no raise


# ── import_catalog: happy path ────────────────────────────────────────────────


class TestImportCatalogHappyPath:
    def test_runs_transaction_in_expected_order(self, mock_conn: MagicMock) -> None:
        mock_conn.sql.return_value.fetchall.return_value = []
        import_ = CatalogExportImport(mock_conn, DB_CONFIG)

        import_.import_catalog("/tmp/backup.duckdb")

        calls = _sql_calls(mock_conn)
        assert calls[0] == "BEGIN"
        assert "ATTACH 'postgres:" in calls[1]
        assert "ATTACH 'duckdb:/tmp/backup.duckdb'" in calls[2]
        assert "COMMIT" in calls
        assert "DETACH from_backup;" in calls
        assert "DETACH restore_db;" in calls

    def test_deletes_and_reinserts_every_backup_table(
        self, mock_conn: MagicMock
    ) -> None:
        mock_conn.sql.return_value.fetchall.return_value = [
            ("table_a",),
            ("table_b",),
        ]
        import_ = CatalogExportImport(mock_conn, DB_CONFIG)

        import_.import_catalog("/tmp/backup.duckdb")

        calls = _sql_calls(mock_conn)
        assert any("DELETE FROM restore_db.my_schema.table_a" in c for c in calls)
        assert any("INSERT INTO restore_db.my_schema.table_a" in c for c in calls)
        assert any("DELETE FROM restore_db.my_schema.table_b" in c for c in calls)
        assert any("INSERT INTO restore_db.my_schema.table_b" in c for c in calls)


# ── import_catalog: failure handling ──────────────────────────────────────────


class TestImportCatalogFailureHandling:
    def test_rolls_back_and_reraises_on_failure(self, mock_conn: MagicMock) -> None:
        def sql_side_effect(query: str, *args: object) -> MagicMock:
            if "ATTACH 'duckdb:" in query:
                raise RuntimeError("boom")
            return MagicMock()

        mock_conn.sql.side_effect = sql_side_effect
        import_ = CatalogExportImport(mock_conn, DB_CONFIG)

        with pytest.raises(RuntimeError, match="boom"):
            import_.import_catalog("/tmp/backup.duckdb")

        assert call("ROLLBACK") in mock_conn.sql.call_args_list

    def test_rollback_failure_is_swallowed(self, mock_conn: MagicMock) -> None:
        """If ROLLBACK itself raises, the original exception must still propagate."""

        def sql_side_effect(query: str, *args: object) -> MagicMock:
            if "ATTACH 'duckdb:" in query:
                raise RuntimeError("boom")
            if query == "ROLLBACK":
                raise RuntimeError("rollback also failed")
            return MagicMock()

        mock_conn.sql.side_effect = sql_side_effect
        import_ = CatalogExportImport(mock_conn, DB_CONFIG)

        with pytest.raises(RuntimeError, match="boom"):
            import_.import_catalog("/tmp/backup.duckdb")


import datetime
from unittest.mock import MagicMock

import gcsfs
import pytest
from gcsfs.retry import HttpError

from ssb_parquedit.connection import DuckDBConnection
from ssb_parquedit.local import LocalDuckDBConnection


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def duck_mock_conn() -> MagicMock:
    conn = MagicMock()
    conn.execute = MagicMock()
    conn.execute.fetchall = MagicMock()
    conn.execute.sql = MagicMock()
    return conn


@pytest.fixture()
def closed_conn(conn: LocalDuckDBConnection) -> LocalDuckDBConnection:
    """A connection that has already been closed."""
    conn.close()
    return conn


# ── CatalogExportImport ──────────────────────────────────────────────────────────────────
class TestCatalogExportImport:
    def test_export_catalog_db_config_is_none(
        self, duck_mock_conn: DuckDBConnection
    ) -> None:
        cei = CatalogExportImport(duck_mock_conn, None)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="db_config is not initialized"):
            cei.export_catalog()

    @patch("ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem")
    @patch("ssb_parquedit.catalogexportimport.datetime.datetime")
    def test_export_catalog_db_config_is_some(
        self,
        mock_time: datetime.datetime,
        _mock_gcfs: gcsfs.GCSFileSystem,
        duck_mock_conn: DuckDBConnection,
    ) -> None:
        db_config = MagicMock()
        cei = CatalogExportImport(duck_mock_conn, db_config)
        time = mock_time.now()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mock_time.now.strftime.return_value = timestamp  # type: ignore[attr-defined]
        schema = f"{db_config['metadata_schema']}"
        expected_output = (
            f"{db_config['data_path']}/catalog-export/{timestamp}_{schema}.duckdb"
        )

        out = cei.export_catalog()
        assert out == expected_output

    @patch("ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem")
    def test_export_catalog_missing_bucket(
        self, mock_gcsfs_cls: gcsfs.GCSFileSystem, duck_mock_conn: DuckDBConnection
    ) -> None:
        mock_gcsfs_cls.return_value.put.side_effect = HttpError({"code": 404})
        db_config = MagicMock()
        cei = CatalogExportImport(duck_mock_conn, db_config)
        with pytest.raises(HttpError):
            cei.export_catalog()

    @patch("ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem")
    def test_export_catalog_no_connection(
        self, _mock_gcsfs: gcsfs.GCSFileSystem, closed_conn: LocalDuckDBConnection
    ) -> None:
        db_config = MagicMock()
        cei = CatalogExportImport(closed_conn, db_config)
        with pytest.raises(RuntimeError):
            cei.export_catalog()
