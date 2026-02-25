# SQL Injection Prevention Implementation

This document describes the SQL injection prevention measures implemented in ssb-parquedit.

## Overview

The codebase has been refactored to use parameterized queries (bind variables) to prevent SQL injection attacks, where applicable.

## Implementation Approach

### 1. Identifier Validation (Table Names, Column Names)

**Status**: ✅ Implemented

DuckDB does not support parameterizing SQL identifiers (table names, column names, schema names). Instead, these are validated using strict whitelist patterns:

- **Table names**: Must start with a letter or underscore, contain only alphanumeric characters and underscores
- **Column names**: Must follow the same pattern as table names
- **Partition columns**: Validated before use in ALTER TABLE statements

**Location**: `utils.py` - `SchemaUtils.validate_table_name()` and `SQLSanitizer.validate_column_list()`

### 2. Parameterized WHERE and ORDER BY Clauses

**Status**: ✅ Partial (Defense-in-depth)

WHERE and ORDER BY clauses are passed as strings because they often contain complex expressions. These cannot be fully parameterized without a major architectural refactor. Instead, the implementation uses:

#### WHERE Clause Validation
- Checks for dangerous SQL keywords that shouldn't appear in WHERE clauses
- Detects SQL comment sequences (`--`, `/*`, `*/`)
- Allows legitimate keywords like `CAST` that are used in normal WHERE expressions

**Location**: `utils.py` - `SQLSanitizer.validate_where_clause()`

#### ORDER BY Clause Validation  
- Stricter validation than WHERE clauses
- Only allows column names, ASC/DESC keywords, and basic arithmetic operators
- Rejects any dangerous SQL patterns
- Uses regex pattern matching to ensure valid format

**Location**: `utils.py` - `SQLSanitizer.validate_order_by_clause()`

**Called from**: `query.py` - `select()` and `count()` methods

### 3. Parameterized Numeric Values

**Status**: ✅ Implemented

LIMIT and OFFSET values are now passed as parameterized values using DuckDB's `?` placeholder syntax.

**Before**:
```python
query += f" LIMIT {limit}"
query += f" OFFSET {offset}"
result = self.conn.execute(query)
```

**After**:
```python
if limit is not None:
    query += " LIMIT ?"
    params.append(limit)
if offset > 0:
    query += " OFFSET ?"
    params.append(offset)
result = self.conn.execute(query, params)
```

**Location**: `query.py` - `select()` method

### 4. Parameterized File Paths

**Status**: ✅ Implemented

File paths passed to `read_parquet()` are now parameterized instead of string interpolated.

**Before**:
```python
ddl = f"""
CREATE TABLE {table_name} AS
SELECT * FROM read_parquet('{parquet_path}')
"""
self.conn.execute(ddl)
```

**After**:
```python
ddl = f"""
CREATE TABLE {table_name} AS
SELECT * FROM read_parquet(?)
"""
self.conn.execute(ddl, [parquet_path])
```

**Locations**: 
- `ddl.py` - `_create_from_parquet()` method
- `dml.py` - `_insert_from_parquet()` method

## Security Considerations

### What's Protected

- Numeric LIMIT/OFFSET values
- File paths in read_parquet()
- Column names in SELECT clauses, partitioning
- Table names

### What's Not Fully Protected (Defense-in-Depth)

- WHERE clause values: These are validated but not fully parameterized due to architectural constraints
- ORDER BY expressions: Validated with strict pattern matching

**Recommendations for Users**:
1. Always validate and sanitize WHERE clause input at the application level
2. Use prepared statements in your application when constructing WHERE clauses
3. Consider using DuckDB's filter APIs or query builders that support parameterization
4. Run ssb-parquedit with database users that have minimal required permissions

## Security Usage Examples

### Safe Usage

```python
# Parameterized LIMIT/OFFSET
df = editor.view("users", limit=10, offset=20)

# Validated table and column names
df = editor.view("users", columns=["id", "name", "email"])

# Structured filters (RECOMMENDED)
df = editor.view("users", filters={"column": "age", "operator": ">", "value": 25})

# Parameterized file paths
editor.insert_data("users", "/path/to/users.parquet")
```

### Things to Avoid

```python
# DON'T construct filter values without parameterization
# Instead, use structured filters which handle parameterization automatically:
user_age = user_input  # Could be malicious
df = editor.view("users", filters={"column": "age", "operator": ">", "value": user_age})

# The value is automatically parameterized, preventing injection
# Even if user_age contains: "25); DROP TABLE users; --"
# It will be safely treated as a literal string value
```

## Testing

To verify the SQL injection prevention measures are working:

```bash
# Run the test suite
pytest tests/

# Run type checking
mypy src/ssb_parquedit/

# Check for potential issues
ruff check src/
```

## Migration Notes

The refactoring is backward compatible - existing code will continue to work, but now with enhanced security:

1. LIMIT/OFFSET now use parameterized queries internally (transparent to users)
2. WHERE/ORDER BY clauses are validated before execution
3. File paths are parameterized (transparent to users)
4. Column and table name validation is more explicit (will raise errors on invalid names)

## References

- [DuckDB Documentation - Parameter Binding](https://duckdb.org/docs/api/overview)
- [OWASP - SQL Injection](https://owasp.org/www-community/attacks/SQL_Injection)
- [CWE-89: SQL Injection](https://cwe.mitre.org/data/definitions/89.html)
