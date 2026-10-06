import unittest
from unittest.mock import MagicMock

from ssb_parquedit.connection import DuckDBConnection
from ssb_parquedit.query import QueryOperations


class TestView(unittest.TestCase):
    def setUp(self) -> None:
        self.conn: DuckDBConnection = MagicMock()
        self.db_config: dict[str, str] = MagicMock()

        self.conn.execute.return_value.df.return_value = "pandas"
        self.conn.execute.return_value.pl.return_value = "polars"
        self.conn.execute.return_value.arrow.return_value = "pyarrow"

        self.qo = QueryOperations(self.conn, self.db_config)

    def test_invalid_output_format(self) -> None:
        table_name = "test_table"
        output_format = "INVALID"
        with self.assertRaises(ValueError) as cm:
            self.qo.view(table_name, output_format=output_format)
        e = cm.exception
        self.assertEqual(e.args[0], f"Unknown output_format: {output_format}. Must be 'pandas', 'polars', or 'pyarrow'.")

    def test_output_format_pandas(self) -> None:
        table_name = "test_table"
        output_format = "pandas"
        result = self.qo.view(table_name, output_format=output_format)
        self.assertEqual(result, "pandas")

    def test_output_format_polars(self) -> None:
        table_name = "test_table"
        output_format = "polars"
        result = self.qo.view(table_name, output_format=output_format)
        self.assertEqual(result, "polars")

    def test_output_format_pyarrow(self) -> None:
        table_name = "test_table"
        output_format = "pyarrow"
        result = self.qo.view(table_name, output_format=output_format)
        self.assertEqual(result, "pyarrow")