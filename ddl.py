"""DDL (Data Definition Language) operations for DuckDB tables."""

from typing import Any
import pandas as pd
from utils import SchemaUtils


class DDLOperations:
    """DDL operations for creating and modifying table structures.
    
    This class handles:
    - Table creation from DataFrames, schemas, or Parquet files
    - Table dropping and alteration
    - Partitioning configuration
    - Table descriptions/comments
    """
    
    def __init__(self, connection):
        """Initialize with a DuckDB connection.
        
        Args:
            connection: DuckDBConnection instance.
        """
        self.conn = connection
    
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
            table_name: Name of the table to create. Must start with a letter or 
                underscore and contain only alphanumeric characters and underscores.
            source: Source for table schema. Can be:
                - pd.DataFrame: Uses DataFrame schema to create table structure
                - dict: JSON Schema specification defining the table structure
                - str: Path to Parquet file (gs:// format) to infer schema from
            table_description: Description/comment for the table. This will be stored
                as a SQL COMMENT on the table.
            part_columns: List of column names to partition by. Defaults to None (no partitioning).
                When specified, the table will be partitioned by these columns for better query performance.
            fill: Whether to populate the table with data from source. Defaults to False.
                Note: This parameter is handled by the facade, not internally in this method.

        Raises:
            ValueError: If table_name contains invalid characters.
            TypeError: If source is not a DataFrame, dict, or string.

        Example:
            >>> # Create from DataFrame
            >>> df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
            >>> ddl.create_table("users", df, "User data")
            
            >>> # Create from JSON Schema
            >>> schema = {
            ...     "properties": {
            ...         "id": {"type": "integer"},
            ...         "name": {"type": "string"}
            ...     },
            ...     "required": ["id"]
            ... }
            >>> ddl.create_table("users", schema, "User data")
            
            >>> # Create from Parquet file with partitioning
            >>> ddl.create_table("events", "gs://bucket/events.parquet", 
            ...                  "Event data", part_columns=["date", "region"])
        """
        SchemaUtils.validate_table_name(table_name)
        
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
        
        self._add_table_description(table_name, table_description)
    
    def drop_table(self, table_name: str) -> None:
        """Drop a table from the catalog.
        
        Args:
            table_name: Name of the table to drop.
            
        Example:
            >>> ddl.drop_table("old_users")
        """
        SchemaUtils.validate_table_name(table_name)
        self.conn.execute(f"DROP TABLE {table_name}")
    
    def alter_table(self, table_name: str, changes: dict[str, Any]) -> None:
        """Alter table structure.
        
        Args:
            table_name: Name of the table to alter.
            changes: Dictionary of changes to apply.
            
        Note:
            Implementation depends on specific alteration requirements.
        """
        SchemaUtils.validate_table_name(table_name)
        # Implementation would depend on specific alteration needs
        raise NotImplementedError("alter_table not yet implemented")
    
    def _create_from_dataframe(self, table_name: str, data: pd.DataFrame) -> None:
        """Create an empty table from a DataFrame schema.
        
        Args:
            table_name: Name of the table to create.
            data: DataFrame whose schema will be used.
        """
        self.conn.register("data", data)
        self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM data WHERE 1=2")
    
    def _create_from_parquet(self, table_name: str, parquet_path: str) -> None:
        """Create an empty table from a Parquet file schema.
        
        Args:
            table_name: Name of the table to create.
            parquet_path: Path to the Parquet file (supports gs:// URIs).
        """
        self.conn.execute(
            f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_parquet('{parquet_path}') WHERE 1=2;
            """
        )
    
    def _create_from_schema(self, table_name: str, schema: dict[str, Any]) -> None:
        """Create a table from a JSON Schema specification.
        
        Args:
            table_name: Name of the table to create.
            schema: JSON Schema dictionary defining the table structure.
        """
        ddl = SchemaUtils.jsonschema_to_duckdb(schema, table_name)
        self.conn.execute(ddl)
    
    def _add_table_description(self, table_name: str, description: str) -> None:
        """Add a comment/description to a table.
        
        Args:
            table_name: Name of the table.
            description: Description text to add as a comment.
        """
        # Escape single quotes in description
        escaped_description = description.replace("'", "''")
        self.conn.execute(f"COMMENT ON TABLE {table_name} IS '{escaped_description}';")
    
    def _add_table_partition(self, table_name: str, part_columns: list[str]) -> None:
        """Configure partitioning for a table.
        
        Args:
            table_name: Name of the table to partition.
            part_columns: List of column names to partition by.
        """
        cols = ",".join(part_columns)
        self.conn.execute(f"ALTER TABLE {table_name} SET PARTITIONED BY ({cols});")
