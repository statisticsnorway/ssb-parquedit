import duckdb
import pandas as pd
import re
import json
import gcsfs

"""
parquedit.py

Utilities for creating and managing DuckDB tables backed by DuckLake storage.

This module exposes the ParquEdit class which provides convenience helpers to:
- attach a DuckLake catalog
- create DuckDB tables from pandas DataFrames, JSON Schema dicts, or Parquet files
- fill tables from DataFrames or Parquet files
- translate JSON Schema types to DuckDB column types
- annotate tables with descriptions and set partitioning columns

Intended usage:
    with ParquEdit(db_config) as p:
        p.create_table("my_table", df, "A table for ...", part_columns=["dt"], fill=True)

Notes:
- This class registers a GCS filesystem (via gcsfs) with DuckDB to allow gs:// reads.
- The class will install and load the DuckLake and PostgreSQL extensions when instantiated.
"""

class ParquEdit:
    """High-level helper for creating and populating DuckDB tables backed by DuckLake.

    The class manages a DuckDB connection (either a provided connection or one created
    internally). It also registers a GCS filesystem and attaches a DuckLake catalog
    according to the provided `db_config`.

    Example:
        db_config = {
            "dbname": "mydb",
            "dbuser": "user",
            "catalog_name": "my_catalog",
            "data_path": "gs://bucket/path",
            "metadata_schema": "ducklake_metadata"
        }
        with ParquEdit(db_config) as pe:
            pe.create_table("events", df, "Event table", part_columns=["date"], fill=True)

    Parameters:
        db_config (dict): Configuration required to attach the DuckLake catalog.
            Required keys: "dbname", "dbuser", "catalog_name", "data_path", "metadata_schema".
        conn (duckdb.DuckDBPyConnection | None): Optional existing DuckDB connection.
            If omitted, a new connection is created and closed when this instance is closed.
    """

    def __init__(
        self,
        db_config: dict,
        conn: duckdb.DuckDBPyConnection | None = None
    ):
        """Initialize ParquEdit, attach DuckLake catalog and prepare connection.

        This will:
        - create or use the provided DuckDB connection
        - register GCS filesystem support so 'read_parquet' can access gs:// paths
        - install and load the 'ducklake' and 'postgres' DuckDB extensions
        - attach the DuckLake catalog and set it as the current catalog

        Raises:
            KeyError: if required keys are missing from db_config (may surface as
                a KeyError when formatting the ATTACH SQL).
            duckdb.Error: if DuckDB extension install/load/attach fails.
        """
        self._owns_conn = conn is None
        self._conn = conn or duckdb.connect()

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

    def __enter__(self):
        """Enter context manager and return this ParquEdit instance."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context manager and close the connection if owned.

        If this instance created the DuckDB connection, it will be closed on exit.
        If an external connection was supplied, it will be left open.
        """
        if self._owns_conn:
            self._conn.close()

    def close(self):
        """Manually close the connection if this instance owns it.

        This is useful when not using the context manager protocol.

        Raises:
            duckdb.Error: if closing the underlying connection fails.
        """
        if self._owns_conn:
            self._conn.close()

    def create_table(
        self,
        table_name: str,
        source: pd.DataFrame | dict | str,
        table_description: str,
        part_columns: list = [],
        fill: bool = False
    ):
        """Create a DuckDB table from a DataFrame, JSON Schema dict, or Parquet path.

        The table is created empty (schema only). Optionally partitioning columns can
        be set and the table can be filled with the provided source.

        Parameters:
            table_name (str): Valid DuckDB table name (alphanumeric + underscores,
                cannot start with a digit).
            source (pd.DataFrame | dict | str): Source to create the table from:
                - pandas.DataFrame: create with the DataFrame's columns
                - dict: JSON Schema dict used to generate DDL
                - str: gs:// path to a Parquet file to infer schema from
            table_description (str): Description to add as table COMMENT.
            part_columns (list): List of column names to partition the table by.
            fill (bool): If True, insert the source data into the newly created table.

        Raises:
            TypeError: if `source` is not an accepted type.
            ValueError: if `table_name` is invalid.
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

        if part_columns:
            self._add_table_partition(table_name,part_columns)    

        if fill:
            self.fill_table(table_name, source)

        self._add_table_description(table_name, table_description)    

    def _create_from_dataframe(self, table_name: str, data: pd.DataFrame):
        """Create an empty table whose schema matches the provided DataFrame.

        The created table will have the same column names and inferred types as `data`,
        but will contain no rows.

        Parameters:
            table_name (str): Target table name.
            data (pd.DataFrame): DataFrame used to infer schema.
        """
        self._conn.register("data", data)
        self._conn.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM data WHERE 1=2"
        )

    def _create_from_parquet(self, table_name: str, parquet_path: str):
        """Create an empty table by reading the schema from a Parquet file.

        Parameters:
            table_name (str): Target table name.
            parquet_path (str): Path to Parquet file (supports gs:// via gcsfs).
        """
        self._conn.execute(
            f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_parquet('{parquet_path}') WHERE 1=2;
            """
        )  

    def _create_from_schema(self, table_name: str, schema: dict):
        """Create a table from a JSON Schema dictionary.

        Parameters:
            table_name (str): Target table name.
            schema (dict): JSON Schema that defines properties and required fields.

        Notes:
            The JSON Schema should have a top-level "properties" mapping. Types are
            translated by jsonschema_to_duckdb and ParquEdit.translate.
        """
        ddl = ParquEdit.jsonschema_to_duckdb(schema, table_name)
        self._conn.execute(ddl)

    def fill_table(self, table_name, source: pd.DataFrame | dict | str):
        """Insert rows into an existing table from a DataFrame or Parquet file.

        Parameters:
            table_name (str): Target table to insert into.
            source (pd.DataFrame | str): DataFrame or gs:// Parquet path containing data.

        Raises:
            TypeError: if `source` is not an accepted type for filling.
        """
        if isinstance(source, pd.DataFrame):
            self._fill_from_dataframe(table_name, data=source)
        elif isinstance(source, str):
            self._fill_from_parquet(table_name, parquet_path=source)

    def _fill_from_dataframe(self, table_name: str, data: pd.DataFrame):
        """Insert rows into `table_name` from a pandas DataFrame.

        Parameters:
            table_name (str): Target table to insert into.
            data (pd.DataFrame): DataFrame with rows to append.
        """
        self._conn.register("data", data)
        self._conn.execute(
            f"INSERT INTO {table_name} SELECT * FROM data"
        )     

    def _fill_from_parquet(self, table_name: str, parquet_path: str):
        """Insert rows into `table_name` by reading from a Parquet file.

        Parameters:
            table_name (str): Target table to insert into.
            parquet_path (str): Path to Parquet file (supports gs:// via gcsfs).
        """
        self._conn.sql(
            f"""
            INSERT INTO {table_name} 
            SELECT * FROM read_parquet('{parquet_path}');
            """
        )                        

    @staticmethod
    def translate(prop):
        """Translate a JSON Schema property to a DuckDB column type.

        Supports JSON Schema primitive types, arrays, and nested objects. If a
        property is a union that includes 'null' (e.g. ["string", "null"]), the
        non-null type is chosen and the NOT NULL constraint is handled separately.

        Parameters:
            prop (dict): A JSON Schema property definition.

        Returns:
            str: A DuckDB column type (e.g., 'VARCHAR', 'BIGINT', 'LIST<DOUBLE>', 'STRUCT(...)').

        Examples:
            translate({"type": "string"}) -> "VARCHAR"
            translate({"type": ["string", "null"], "format": "date"}) -> "DATE"
            translate({"type": "array", "items": {"type": "integer"}}) -> "LIST<BIGINT>"
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
    def jsonschema_to_duckdb(schema, table_name):
        """Convert a JSON Schema to a DuckDB CREATE TABLE statement.

        Parameters:
            schema (dict): JSON Schema with a top-level 'properties' dict and optional 'required' list.
            table_name (str): Desired table name for the generated CREATE TABLE statement.

        Returns:
            str: A CREATE TABLE SQL statement representing the schema.

        Notes:
            - Required properties are annotated with NOT NULL.
            - Complex nested objects are translated to STRUCT types where possible.
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
    def _validate_table_name(table_name: str):
        """Validate that `table_name` is a safe SQL identifier for DuckDB.

        Allowed pattern: starts with a letter or underscore, followed by letters, digits, or underscores.

        Raises:
            ValueError: If the table name does not match the allowed pattern.
        """
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(f"Invalid table name: {table_name}")

    def _add_table_description(self, table_name: str, description: str):
        """Add a COMMENT to the specified table.

        Parameters:
            table_name (str): Table to annotate.
            description (str): Text to set as the table comment.

        Notes:
            This issues a `COMMENT ON TABLE ... IS '...'` SQL command.
        """
        self._conn.execute(
            f"COMMENT ON TABLE {table_name} IS '{description}';"
        )  

    def _add_table_partition(self, table_name: str, part_columns: list):
        """Set partitioning columns for a DuckLake-backed table.

        Parameters:
            table_name (str): Table to alter.
            part_columns (list): List of column names to partition by.

        Notes:
            This issues an `ALTER TABLE ... SET PARTITIONED BY (...)` command. The
            columns must exist on the table or the SQL will fail.
        """
        cols = ",".join(part_columns)
        self._conn.execute(
            f"ALTER TABLE {table_name} SET PARTITIONED BY ({cols});"
        ) 
      
