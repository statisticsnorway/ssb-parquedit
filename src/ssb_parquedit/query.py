"""Query operations for DuckDB tables."""

import pandas as pd
from utils import SchemaUtils


class QueryOperations:
    """Query operations for reading and analyzing table data.
    
    This class handles:
    - Table data retrieval with filtering, sorting, and pagination
    - Row counting
    - Table existence checks
    """
    
    def __init__(self, connection):
        """Initialize with a DuckDB connection.
        
        Args:
            connection: DuckDBConnection instance.
        """
        self.conn = connection
    
    def view_table(
        self,
        table_name: str,
        limit: int | None = 10,
        offset: int = 0,
        columns: list[str] | None = None,
        where: str | None = None,
        order_by: str | None = None,
    ) -> pd.DataFrame:
        """View contents of a table in the DuckLake catalog.
        
        Args:
            table_name: Name of the table to view.
            limit: Maximum number of rows to return. None returns all rows. Defaults to 10.
            offset: Number of rows to skip. Defaults to 0. Useful for pagination.
            columns: List of column names to select. None selects all columns (*).
            where: WHERE clause condition (without the WHERE keyword). 
                Example: "age > 25" or "status = 'active'"
            order_by: ORDER BY clause (without the ORDER BY keyword).
                Example: "created_at DESC" or "name ASC, age DESC"
        
        Returns:
            pd.DataFrame: DataFrame containing the query results.
        
        Example:
            >>> # Simple view - first 5 rows
            >>> query.view_table("users", limit=5)
            
            >>> # Select specific columns
            >>> query.view_table("users", columns=["id", "name"], limit=10)
            
            >>> # Filter with WHERE clause
            >>> query.view_table("users", where="age > 25", limit=100)
            
            >>> # Sort results
            >>> query.view_table("users", order_by="created_at DESC", limit=10)
            
            >>> # Pagination
            >>> query.view_table("users", limit=10, offset=20)  # Page 3
            
            >>> # Get all rows (no limit)
            >>> query.view_table("users", limit=None)
        """
        SchemaUtils.validate_table_name(table_name)
        
        # Build SELECT clause
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"
        
        # Build query
        query = f"SELECT {select_clause} FROM {table_name}"
        
        # Add WHERE clause
        if where:
            query += f" WHERE {where}"
        
        # Add ORDER BY clause
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # Add LIMIT and OFFSET
        if limit is not None:
            query += f" LIMIT {limit}"
        if offset > 0:
            query += f" OFFSET {offset}"
        
        # Execute and return as DataFrame
        return self.conn.execute(query).df()
    
    def select(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Select data from a table (simplified query interface).
        
        This is a simpler alternative to view_table for basic queries.
        
        Args:
            table_name: Name of the table.
            columns: List of column names to select. None selects all columns.
            where: WHERE clause condition (without the WHERE keyword).
            limit: Maximum number of rows to return. None returns all rows.
        
        Returns:
            pd.DataFrame: Query results.
        
        Example:
            >>> # Select all columns
            >>> query.select("users")
            
            >>> # Select specific columns with filter
            >>> query.select("users", columns=["id", "name"], where="age > 25")
            
            >>> # Limit results
            >>> query.select("users", limit=10)
        """
        return self.view_table(
            table_name=table_name,
            columns=columns,
            where=where,
            limit=limit
        )
    
    def count(self, table_name: str, where: str | None = None) -> int:
        """Count rows in a table.
        
        Args:
            table_name: Name of the table.
            where: Optional WHERE clause condition (without the WHERE keyword).
        
        Returns:
            int: Number of rows matching the condition.
        
        Example:
            >>> # Count all rows
            >>> total = query.count("users")
            
            >>> # Count with filter
            >>> active = query.count("users", where="status = 'active'")
            
            >>> # Count with complex condition
            >>> recent = query.count("users", where="created_at > '2024-01-01' AND age > 18")
        """
        SchemaUtils.validate_table_name(table_name)
        
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        if where:
            query += f" WHERE {where}"
        
        result = self.conn.execute(query).df()
        return int(result['count'].iloc[0])
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the catalog.
        
        Args:
            table_name: Name of the table to check.
        
        Returns:
            bool: True if table exists, False otherwise.
        
        Example:
            >>> if query.table_exists("users"):
            ...     print("Table exists")
        """
        try:
            self.conn.execute(
                f"SELECT 1 FROM {table_name} WHERE 1=0"
            )
            return True
        except Exception:
            return False
