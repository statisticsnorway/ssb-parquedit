

import unittest
from unittest.mock import MagicMock

import pytest

from ssb_parquedit.maintenance import MaintenanceOperations


class TestMaintenanceOperations(unittest.TestCase):
    def test_flush_inlined_table_invalid_table_name(self) -> None:
        conn = MagicMock()
        dbconfig = MagicMock()

        table_name = "INVALID"

        mo = MaintenanceOperations(conn, dbconfig)

        with pytest.raises(ValueError):
            mo.flush_inlined_table(table_name)

    def test_flush_inlined_table_db_config_is_none(self) -> None:
        conn = MagicMock()
        dbconfig = None

        table_name = "valid"

        mo = MaintenanceOperations(conn, dbconfig)

        with pytest.raises(RuntimeError, match="db_config is not initialized"):
            mo.flush_inlined_table(table_name)

    def test_flushed_inlined_table_rows_flushed(self) -> None:
        conn = MagicMock()
        dbconfig = MagicMock()

        conn.execute.return_value.fetchall.return_value = [
            ["test", "test", 2],
            ["test2", "test2", 6]
        ]

        table_name = "valid"

        mo = MaintenanceOperations(conn, dbconfig)

        with self.assertLogs("ssb_parquedit.maintenance", level="INFO") as cm:
            mo.flush_inlined_table(table_name)
            self.assertEqual(cm.output, [f"INFO:ssb_parquedit.maintenance:Flushed 8 rows for table '{table_name}'.",])

    def test_flushed_inlined_table_no_rows_flushed(self) -> None:
        conn = MagicMock()
        dbconfig = MagicMock()

        conn.execute.return_value.fetchall.return_value = []

        table_name = "valid"
        
        mo = MaintenanceOperations(conn, dbconfig)

        with self.assertLogs("ssb_parquedit.maintenance", level="INFO") as cm:
            mo.flush_inlined_table(table_name)
            self.assertEqual(cm.output, [f"INFO:ssb_parquedit.maintenance:No inlined data to flush for table '{table_name}'.",])

    def test_merge_adjacent_files_db_config_is_none(self) -> None:
        conn = MagicMock()
        dbconfig = None

        mo = MaintenanceOperations(conn, dbconfig)

        table_name = "valid"

        with self.assertRaises(RuntimeError) as cm:
            mo.merge_adjacent_files(table_name)
        
        the_exception: RuntimeError = cm.exception
        self.assertEqual(the_exception.args[0], "db_config is not initialized")

    def test_merge_adjacent_files_some_files_merged(self) -> None:
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            ["s1", "t1", 6, 4],
            ["s2", "t2", 1, 5],
        ]

        dbconfig = MagicMock()

        mo = MaintenanceOperations(conn, dbconfig)

        table_name = "valid"

        with self.assertLogs("ssb_parquedit.maintenance", level="INFO") as cm:
            mo.merge_adjacent_files(table_name)
            self.assertEqual(cm.output, [f"INFO:ssb_parquedit.maintenance:Merged 7 files into 9 for table '{table_name}'.",])

    def test_merge_adjacent_files_no_files_merged(self) -> None:
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = []

        dbconfig = MagicMock()

        mo = MaintenanceOperations(conn, dbconfig)

        table_name = "valid"

        with self.assertLogs("ssb_parquedit.maintenance", level="INFO") as cm:
            mo.merge_adjacent_files(table_name)
            self.assertEqual(cm.output, [f"INFO:ssb_parquedit.maintenance:No files to merge for table '{table_name}'.",])