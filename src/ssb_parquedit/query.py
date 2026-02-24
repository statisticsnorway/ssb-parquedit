"""Query operations for DuckDB tables."""

from typing import Literal, Any
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pyarrow as pa
except ImportError:
    pa = None

from utils import SchemaUtils, SQLSanitizer


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
    
    def select(
        self,
        table_name: str,
        limit: int | None = 10,
        offset: int = 0,
        columns: list[str] | None = None,
        where: str | None = None,
        filters: dict[str, Any] | list[dict[str, Any]] | None = None,
        order_by: str | None = None,
        output_format: Literal["pandas", "polars", "pyarrow"] = "pandas",
    ) -> Any:
        """View contents of a table in the DuckLake catalog.
        
        Args:
            table_name: Name of the table to view.
            limit: Maximum number of rows to return. None returns all rows. Defaults to 10.
            offset: Number of rows to skip. Defaults to 0. Useful for pagination.
            columns: List of column names to select. None selects all columns (*).
            where: WHERE clause condition (without the WHERE keyword). 
                DEPRECATED: Use filters parameter for better security.
                Example: "age > 25" or "status = 'active'"
            filters: Structured filter conditions (preferred over where). Can be:
                - List of dicts: [{"column": "age", "operator": ">", "value": 25}, ...]
                - Dict with 'and'/'or': {"and": [condition1, condition2]}
                - Single dict condition: {"column": "age", "operator": ">", "value": 25}
                Operators supported: =, !=, <>, <, >, <=, >=, LIKE, IN, NOT IN, 
                BETWEEN, IS NULL, IS NOT NULL
            order_by: ORDER BY clause (without the ORDER BY keyword).
                Example: "created_at DESC" or "name ASC, age DESC"
                Only column names and ASC/DESC keywords are allowed.
            output_format: Format for the returned data. Options are:
                - "pandas" (default): Returns pd.DataFrame
                - "polars": Returns pl.DataFrame (requires polars library)
                - "pyarrow": Returns pa.Table (requires pyarrow library)
        
        Returns:
            Data in the specified format (pandas DataFrame, polars DataFrame, or pyarrow Table).
        
        Example:
            >>> # Simple view - first 5 rows as pandas DataFrame
            >>> query.select("users", limit=5)
            
            >>> # Select specific columns
            >>> query.select("users", columns=["id", "name"], limit=10)
            
            >>> # Filter with structured filters (RECOMMENDED)
            >>> query.select("users", filters={"column": "age", "operator": ">", "value": 25})
            
            >>> # Multiple conditions with AND
            >>> query.select("users", filters=[
            ...     {"column": "age", "operator": ">", "value": 25},
            ...     {"column": "status", "operator": "=", "value": "active"}
            ... ])
            
            >>> # Multiple conditions with OR
            >>> query.select("users", filters={
            ...     "or": [
            ...         {"column": "status", "operator": "=", "value": "admin"},
            ...         {"column": "status", "operator": "=", "value": "moderator"}
            ...     ]
            ... })
            
            >>> # Filter with LIKE operator
            >>> query.select("users", filters={"column": "name", "operator": "LIKE", "value": "%john%"})
            
            >>> # Filter with IN operator
            >>> query.select("users", filters={"column": "id", "operator": "IN", "value": [1, 2, 3]})
            
            >>> # Filter with BETWEEN
            >>> query.select("users", filters={"column": "age", "operator": "BETWEEN", "value": [18, 65]})
            
            >>> # Sort results
            >>> query.select("users", order_by="created_at DESC", limit=10)
            
            >>> # Pagination
            >>> query.select("users", limit=10, offset=20)  # Page 3
            
            >>> # Get all rows (no limit)
            >>> query.select("users", limit=None)
            
            >>> # Return as polars DataFrame
            >>> query.select("users", limit=10, output_format="polars")
            
            >>> # Return as pyarrow Table
            >>> query.select("users", limit=10, output_format="pyarrow")
        """
        SchemaUtils.validate_table_name(table_name)
        
        # Validate and sanitize SQL clauses to prevent injection
        SQLSanitizer.validate_order_by_clause(order_by)
        if columns:
            SQLSanitizer.validate_column_list(columns)
        
        # Use filters if provided (recommended), otherwise use where clause
        if filters is not None:
            where_parameterized, where_params = SQLSanitizer.build_where_from_filters(filters)
        else:
            # Fall back to where clause (for backward compatibility)
            where_parameterized, where_params = SQLSanitizer.parameterize_where_clause(where)
        
        # Build SELECT clause
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"
        
        # Build query with parameterized LIMIT and OFFSET
        query = f"SELECT {select_clause} FROM {table_name}"
        params: list[Any] = []
        
        # Add WHERE clause (now parameterized)
        if where_parameterized:
            query += f" WHERE {where_parameterized}"
            params.extend(where_params)
        
        # Add ORDER BY clause
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # Add LIMIT and OFFSET with parameter binding
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        if offset > 0:
            query += " OFFSET ?"
            params.append(offset)
        
        # Execute with parameterized values
        if params:
            result = self.conn.execute(query, params)
        else:
            result = self.conn.execute(query)
        
        # Convert to requested format
        if output_format == "pandas":
            return result.df()
        elif output_format == "polars":
            if pl is None:
                raise ImportError("polars is not installed. Install it with: pip install polars")
            return result.pl()
        elif output_format == "pyarrow":
            if pa is None:
                raise ImportError("pyarrow is not installed. Install it with: pip install pyarrow")
            return result.arrow()
        else:
            raise ValueError(f"Unknown output_format: {output_format}. Must be 'pandas', 'polars', or 'pyarrow'.")
    
    
    def count(self, table_name: str, where: str | None = None, filters: dict[str, Any] | list[dict[str, Any]] | None = None) -> int:
        """Count rows in a table.
        
        Args:
            table_name: Name of the table.
            where: Optional WHERE clause condition (without the WHERE keyword).
                   DEPRECATED: Use filters parameter for better security.
            filters: Structured filter conditions (preferred over where). Can be:
                - List of dicts: [{"column": "age", "operator": ">", "value": 25}, ...]
                - Dict with 'and'/'or': {"and": [condition1, condition2]}
                - Single dict condition: {"column": "age", "operator": ">", "value": 25}
        
        Returns:
            int: Number of rows matching the condition.
        
        Example:
            >>> # Count all rows
            >>> total = query.count("users")
            
            >>> # Count with structured filters (RECOMMENDED)
            >>> active = query.count("users", filters={"column": "status", "operator": "=", "value": "active"})
            
            >>> # Count with complex filters
            >>> recent = query.count("users", filters=[
            ...     {"column": "created_at", "operator": ">", "value": "2024-01-01"},
            ...     {"column": "age", "operator": ">", "value": 18}
            ... ])
        """
        SchemaUtils.validate_table_name(table_name)
        
        # Use filters if provided (recommended), otherwise use where clause
        if filters is not None:
            where_parameterized, where_params = SQLSanitizer.build_where_from_filters(filters)
        else:
            where_parameterized, where_params = SQLSanitizer.parameterize_where_clause(where)
        
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        if where_parameterized:
            query += f" WHERE {where_parameterized}"
        
        result = self.conn.execute(query, where_params).df() if where_params else self.conn.execute(query).df()
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
        # Validate table name first
        SchemaUtils.validate_table_name(table_name)
        
        try:
            self.conn.execute(
                f"SELECT 1 FROM {table_name} WHERE 1=0"
            )
            return True
        except Exception:
            return False
