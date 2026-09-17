import pytest
import datetime
import gcsfs

from gcsfs.retry import HttpError

from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import Mock
from ssb_parquedit.catalogexportimport import CatalogExportImport
from ssb_parquedit.connection import DuckDBConnection
from ssb_parquedit.local import LocalDuckDBConnection

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_conn() -> MagicMock:
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
    def test_export_catalog_db_config_is_none(self, mock_conn: DuckDBConnection) -> None:
        cei = CatalogExportImport(mock_conn, None)
        with pytest.raises(RuntimeError, match="db_config is not initialized"):
            cei.export_catalog()

    @patch('ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem')
    @patch('ssb_parquedit.catalogexportimport.datetime.datetime')
    def test_export_catalog_db_config_is_some(self, mock_time: datetime.datetime, _mock_gcfs: gcsfs.GCSFileSystem,  mock_conn: DuckDBConnection) -> None:
        db_config = MagicMock()
        cei = CatalogExportImport(mock_conn, db_config)
        time = mock_time.now()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mock_time.now.strftime.return_value = timestamp
        schema = f"{db_config['metadata_schema']}"
        expected_output = f"{db_config['data_path']}/catalog-export/{timestamp}_{schema}.duckdb"

        out = cei.export_catalog()
        assert out == expected_output

    def test_export_catalog_missing_bucket(self, mock_conn: DuckDBConnection):
        db_config = MagicMock()
        cei = CatalogExportImport(mock_conn, db_config)
        with pytest.raises(HttpError):
            cei.export_catalog()

    @patch('ssb_parquedit.catalogexportimport.gcsfs.GCSFileSystem')
    def test_export_catalog_no_connection(self, _mock_gcsfs: gcsfs.GCSFileSystem, closed_conn: LocalDuckDBConnection):
        db_config = MagicMock()
        cei = CatalogExportImport(closed_conn, db_config)
        with pytest.raises(RuntimeError):
            cei.export_catalog()