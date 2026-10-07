import unittest
from unittest.mock import patch

from ssb_parquedit.local_backup import LocalCatalogGCSDataConnection


class TestInit(unittest.TestCase):
    def test_catalog_name_is_none(self) -> None:
        catalog_name = None
        catalog_path = "/test/test"
        with (
            patch("ssb_parquedit.local_backup.duckdb"),
            patch("ssb_parquedit.local_backup.gcsfs"),
        ):
            conn = LocalCatalogGCSDataConnection(catalog_path, catalog_name)
        self.assertEqual(conn.catalog_name, "restored_catalog")
        self.assertEqual(conn.catalog_path, catalog_path)

    def test_catalog_name_is_some(self) -> None:
        catalog_name = "test"
        catalog_path = "/test/test"
        with (
            patch("ssb_parquedit.local_backup.duckdb"),
            patch("ssb_parquedit.local_backup.gcsfs"),
        ):
            conn = LocalCatalogGCSDataConnection(catalog_path, catalog_name)
        self.assertEqual(conn.catalog_name, catalog_name)
        self.assertEqual(conn.catalog_path, catalog_path)
