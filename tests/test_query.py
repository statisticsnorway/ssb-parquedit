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


class TestTimeTravel(unittest.TestCase):
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
        at_time = "2024-01-01"
        with self.assertRaises(ValueError) as cm:
            self.qo.time_travel(table_name, at_time=at_time, output_format=output_format)
        e = cm.exception
        self.assertEqual(e.args[0], f"Unknown output_format: {output_format}. Must be 'pandas', 'polars', or 'pyarrow'.")

    def test_columns_is_set(self) -> None:
        ex = MagicMock()
        self.conn.execute = ex
        table_name = "test_table"
        at_time = "2024-01-01"
        columns = ["id", "name"]
        self.qo.time_travel(table_name, at_time=at_time, columns=columns)
        ex_call_args = [c.args[0] for c in ex.call_args_list]
        self.assertIn("rowid, " + ", ".join(columns), ex_call_args[0])

    def test_order_by_is_set(self) -> None:
        ex = MagicMock()
        self.conn.execute = ex
        table_name = "test_table"
        at_time = "2024-01-01"
        order_by = "id"
        self.qo.time_travel(table_name, at_time=at_time, order_by=order_by)
        ex_call_args = [c.args[0] for c in ex.call_args_list]
        self.assertIn("ORDER BY id", ex_call_args[0])

    def test_offset_is_set(self) -> None:
        ex = MagicMock()
        self.conn.execute = ex
        table_name = "test_table"
        at_time = "2024-01-01"
        offset = 10
        self.qo.time_travel(table_name, at_time=at_time, offset=offset)
        ex_call_args = [c.args[0] for c in ex.call_args_list]
        self.assertIn("OFFSET 10", ex_call_args[0])

    def test_output_format_is_pandas(self) -> None:
        table_name = "test_table"
        output_format = "pandas"
        at_time = "2024-01-01"
        result = self.qo.time_travel(table_name, at_time=at_time, output_format=output_format)
        self.assertEqual(result, "pandas")

    def test_output_format_is_polars(self) -> None:
        table_name = "test_table"
        output_format = "polars"
        at_time = "2024-01-01"
        result = self.qo.time_travel(table_name, at_time=at_time, output_format=output_format)
        self.assertEqual(result, "polars")

    def test_output_format_is_pyarrow(self) -> None:
        table_name = "test_table"
        output_format = "pyarrow"
        at_time = "2024-01-01"
        result = self.qo.time_travel(table_name, at_time=at_time, output_format=output_format)
        self.assertEqual(result, "pyarrow")