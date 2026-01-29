import re
from types import TracebackType
from typing import Any

import duckdb
import gcsfs
import pandas as pd


class ParquEdit:
    """A class for managing DuckDB tables with DuckLake catalog integration.

    This class provides an interface for creating and managing tables in DuckDB
    with support for Google Cloud Storage, PostgreSQL metadata, and DuckLake catalogs.


    """

    def __init__(
        self, db_config: dict[str, str], conn: duckdb.DuckDBPyConnection | None = None
    ) -> None:
        """Initialize ParquEdit.

        Instance Attributes:
            _owns_conn (bool): Whether this instance owns the DuckDB connection.
            _conn (duckdb.DuckDBPyConnection): The active DuckDB connection.

        """
        self._owns_conn: bool = conn is None
        self._conn: duckdb.DuckDBPyConnection = conn or duckdb.connect()
        fs = gcsfs.GCSFileSystem()
        self._conn.register_filesystem(fs)
        # Load extensions
        for ext in ("ducklake", "postgres"):
            self._conn.sql(f"INSTALL {ext}")
            self._conn.sql(f"LOAD {ext}")
        # Attach catalog
        self._conn.sql(f"""
            ATTACH 'ducklake:postgres:
                dbname={db_config["dbname"]}
                user={db_config["dbuser"]}
                host=localhost
            ' AS {db_config["catalog_name"]}
            (DATA_PATH '{db_config["data_path"]}',
             METADATA_SCHEMA {db_config["metadata_schema"]});
        """)
        self._conn.sql(f"USE {db_config['catalog_name']}")

    def __enter__(self) -> "ParquEdit":
        """Context manager entry.

        Returns:
            ParquEdit: This instance for use in with statements.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Context manager exit, closes connection if owned.

        Args:
            exc_type: Exception type if an exception occurred.
            exc: Exception instance if an exception occurred.
            tb: Traceback if an exception occurred.
        """
        if self._owns_conn:
            self._conn.close()

    def close(self) -> None:
        """Manually close the connection if this instance owns it."""
        if self._owns_conn:
            self._conn.close()

    def create_table(
        self,
        table_name: str,
        source: pd.DataFrame | dict[str, Any] | str,
        table_description: str,
        part_columns: list[str] | None = None,
        fill: bool = False,
    ) -> None:
        """Create a new table in the DuckLake catalog.

        Args:
            table_name: Name of the table to create.
            source: Source for table schema. Can be:
                - pd.DataFrame: Uses DataFrame schema
                - dict: JSON Schema specification
                - str: Path to Parquet file (gs:// format)
            table_description: Description/comment for the table.
            part_columns: List of column names to partition by. Defaults to [].
            fill: Whether to populate the table with data from source. Defaults to False.

        Raises:
            TypeError: If source is not a DataFrame, dict, or string.
        """
        self._validate_table_name(table_name)
        if isinstance(source, pd.DataFrame):
            self._create_from_dataframe(table_name, source)
        elif isinstance(source, dict):
            self._create_from_schema(table_name, source)
        elif isinstance(source, str):
            self._create_from_parquet(table_name, source)
        else:
            raise TypeError(
                "source must be a DataFrame, JSON Schema dict, or gs:// Parquet path"
            )
        if part_columns is None:
            part_columns = []
        if len(part_columns) > 0:
            self._add_table_partition(table_name, part_columns)
        if fill:
            self.fill_table(table_name, source)
        self._add_table_description(table_name, table_description)

    def _create_from_dataframe(self, table_name: str, data: pd.DataFrame) -> None:
        """Create an empty table from a DataFrame schema.

        Args:
            table_name: Name of the table to create.
            data: DataFrame whose schema will be used.
        """
        self._conn.register("data", data)
        self._conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM data WHERE 1=2")

    def _create_from_parquet(self, table_name: str, parquet_path: str) -> None:
        """Create an empty table from a Parquet file schema.

        Args:
            table_name: Name of the table to create.
            parquet_path: Path to the Parquet file (supports gs:// URIs).
        """
        self._conn.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_parquet('{parquet_path}') WHERE 1=2;
            """)

    def _create_from_schema(self, table_name: str, schema: dict[str, Any]) -> None:
        """Create a table from a JSON Schema specification.

        Args:
            table_name: Name of the table to create.
            schema: JSON Schema dictionary defining the table structure.
        """
        ddl = ParquEdit.jsonschema_to_duckdb(schema, table_name)
        self._conn.execute(ddl)

    def fill_table(
        self, table_name: str, source: pd.DataFrame | dict[str, Any] | str
    ) -> None:
        """Populate an existing table with data.

        Args:
            table_name: Name of the table to fill.
            source: Data source. Can be:
                - pd.DataFrame: Insert DataFrame rows
                - str: Path to Parquet file (gs:// format)
        """
        if isinstance(source, pd.DataFrame):
            self._fill_from_dataframe(table_name, data=source)
        elif isinstance(source, str):
            self._fill_from_parquet(table_name, parquet_path=source)

    def _fill_from_dataframe(self, table_name: str, data: pd.DataFrame) -> None:
        """Insert data from a DataFrame into a table.

        Args:
            table_name: Name of the table to populate.
            data: DataFrame containing the data to insert.
        """
        self._conn.register("data", data)
        self._conn.execute(f"INSERT INTO {table_name} SELECT * FROM data")

    def _fill_from_parquet(self, table_name: str, parquet_path: str) -> None:
        """Insert data from a Parquet file into a table.

        Args:
            table_name: Name of the table to populate.
            parquet_path: Path to the Parquet file (supports gs:// URIs).
        """
        self._conn.sql(f"""
            INSERT INTO {table_name}
            SELECT * FROM read_parquet('{parquet_path}');
            """)

    @staticmethod
    def translate(prop: dict[str, Any]) -> str:
        """Translate a JSON Schema property to a DuckDB column type.

        Args:
            prop: JSON Schema property definition dictionary.

        Returns:
            str: DuckDB column type specification.
        """
        t = prop.get("type")
        if isinstance(t, list):
            # Remove 'null' from union type
            t = next(x for x in t if x != "null")
        if t == "string":
            fmt = prop.get("format")
            if fmt == "date-time":
                return "TIMESTAMP"
            if fmt == "date":
                return "DATE"
            return "VARCHAR"
        if t == "integer":
            return "BIGINT"
        if t == "number":
            return "DOUBLE"
        if t == "boolean":
            return "BOOLEAN"
        if t == "array":
            return f"LIST<{ParquEdit.translate(prop['items'])}>"
        if t == "object":
            props = prop.get("properties")
            if not props:
                return "JSON"
            fields = [f"{k} {ParquEdit.translate(v)}" for k, v in props.items()]
            return f"STRUCT({', '.join(fields)})"
        return "JSON"

    @staticmethod
    def jsonschema_to_duckdb(schema: dict[str, Any], table_name: str) -> str:
        """Convert a JSON Schema to a DuckDB CREATE TABLE statement.

        Args:
            schema: JSON Schema dictionary with 'properties' and optional 'required' fields.
            table_name: Name for the table in the CREATE statement.

        Returns:
            str: DuckDB CREATE TABLE DDL statement.
        """
        required = set(schema.get("required", []))
        cols = []
        for name, prop in schema["properties"].items():
            col = f"{name} {ParquEdit.translate(prop)}"
            if name in required:
                col += " NOT NULL"
            cols.append(col)
        return f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(cols) + "\n);"

    @staticmethod
    def _validate_table_name(table_name: str) -> None:
        """Validate that a table name follows DuckDB naming conventions.

        Args:
            table_name: The table name to validate.

        Raises:
            ValueError: If the table name contains invalid characters.
        """
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(f"Invalid table name: {table_name}")

    def _add_table_description(self, table_name: str, description: str) -> None:
        """Add a comment/description to a table.

        Args:
            table_name: Name of the table.
            description: Description text to add as a comment.
        """
        self._conn.execute(f"COMMENT ON TABLE {table_name} IS '{description}';")

    def _add_table_partition(self, table_name: str, part_columns: list[str]) -> None:
        """Configure partitioning for a table.

        Args:
            table_name: Name of the table to partition.
            part_columns: List of column names to partition by.
        """
        cols = ",".join(part_columns)
        self._conn.execute(f"ALTER TABLE {table_name} SET PARTITIONED BY ({cols});")
