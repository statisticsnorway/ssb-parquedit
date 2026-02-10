"""DML (Data Manipulation Language) operations for DuckDB tables."""

from typing import Any
import pandas as pd


class DMLOperations:
    """DML operations for inserting, updating, and deleting table data.
    
    This class handles:
    - Data insertion from DataFrames or Parquet files
    - Row updates with filtering
    - Row deletions with filtering
    """
    
    def __init__(self, connection):
        """Initialize with a DuckDB connection.
        
        Args:
            connection: DuckDBConnection instance.
        """
        self.conn = connection
    
    def fill_table(
        self, table_name: str, source: pd.DataFrame | dict[str, Any] | str
    ) -> None:
        """Populate an existing table with data.

        Args:
            table_name: Name of the table to fill.
            source: Data source. Can be:
                - pd.DataFrame: Insert DataFrame rows into the table
                - str: Path to Parquet file (gs:// format) to read and insert data from

        Raises:
            TypeError: If source is not a DataFrame or string.

        Example:
            >>> # Fill from DataFrame
            >>> df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
            >>> dml.fill_table("users", df)
            
            >>> # Fill from Parquet file
            >>> dml.fill_table("users", "gs://bucket/users.parquet")
        """
        if isinstance(source, pd.DataFrame):
            self._fill_from_dataframe(table_name, data=source)
        elif isinstance(source, str):
            self._fill_from_parquet(table_name, parquet_path=source)
        else:
            raise TypeError("source must be a DataFrame or gs:// Parquet path")
    
    def insert(self, table_name: str, data: pd.DataFrame) -> None:
        """Insert data into a table.
        
        Args:
            table_name: Name of the table.
            data: DataFrame containing rows to insert.
            
        Example:
            >>> df = pd.DataFrame({"id": [3, 4], "name": ["Charlie", "Diana"]})
            >>> dml.insert("users", df)
        """
        self._fill_from_dataframe(table_name, data)
    
    def insert_from_parquet(self, table_name: str, parquet_path: str) -> None:
        """Insert data from a Parquet file into a table.
        
        Args:
            table_name: Name of the table.
            parquet_path: Path to Parquet file (supports gs:// URIs).
            
        Example:
            >>> dml.insert_from_parquet("users", "gs://bucket/new_users.parquet")
        """
        self._fill_from_parquet(table_name, parquet_path)
    
    def update(self, table_name: str, updates: dict[str, Any], where: str) -> None:
        """Update rows in a table.
        
        Args:
            table_name: Name of the table.
            updates: Dictionary of column: new_value pairs to update.
            where: WHERE clause condition (without the WHERE keyword).
            
        Example:
            >>> # Update single column
            >>> dml.update("users", {"status": "active"}, "id = 1")
            
            >>> # Update multiple columns
            >>> dml.update("users", 
            ...            {"status": "active", "updated_at": "2024-01-01"}, 
            ...            "age > 25")
        """
        # Build SET clause
        set_items = []
        for col, val in updates.items():
            if isinstance(val, str):
                set_items.append(f"{col} = '{val}'")
            elif val is None:
                set_items.append(f"{col} = NULL")
            else:
                set_items.append(f"{col} = {val}")
        
        set_clause = ", ".join(set_items)
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {where}"
        self.conn.execute(sql)
    
    def delete(self, table_name: str, where: str) -> None:
        """Delete rows from a table.
        
        Args:
            table_name: Name of the table.
            where: WHERE clause condition (without the WHERE keyword).
            
        Example:
            >>> # Delete specific rows
            >>> dml.delete("users", "age < 18")
            
            >>> # Delete with complex condition
            >>> dml.delete("users", "status = 'inactive' AND last_login < '2023-01-01'")
        """
        sql = f"DELETE FROM {table_name} WHERE {where}"
        self.conn.execute(sql)
    
    def _fill_from_dataframe(self, table_name: str, data: pd.DataFrame) -> None:
        """Insert data from a DataFrame into a table.
        
        Args:
            table_name: Name of the table to populate.
            data: DataFrame containing the data to insert.
        """
        self.conn.register("data", data)
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM data")
    
    def _fill_from_parquet(self, table_name: str, parquet_path: str) -> None:
        """Insert data from a Parquet file into a table.
        
        Args:
            table_name: Name of the table to populate.
            parquet_path: Path to the Parquet file (supports gs:// URIs).
        """
        self.conn.sql(
            f"""
            INSERT INTO {table_name}
            SELECT * FROM read_parquet('{parquet_path}');
            """
        )
