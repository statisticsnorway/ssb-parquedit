# Structured Filters and Parameterized Queries

This document describes the new structured filtering capabilities in ssb-parquedit for safe, parameterized query building.

## Overview

The `select()` and `count()` methods now support a new `filters` parameter that provides a structured, type-safe way to specify query conditions. This completely eliminates SQL injection risk for WHERE clause values while maintaining full flexibility.

## Why Structured Filters?

### Problems with String-based WHERE Clauses
```python
# ❌ Vulnerable to SQL injection
user_input = "25); DROP TABLE users; --"
df = editor.select("users", where=f"age > {user_input}")
# Executes: SELECT * FROM users WHERE age > 25); DROP TABLE users; --
```

### Solution: Structured Filters
```python
# ✅ Safe - values are parameterized
df = editor.select("users", filters={"column": "age", "operator": ">", "value": 25})
# All values are passed as parameters, not concatenated into SQL
```

## Filter Syntax

### Basic Single Condition

```python
# Simple comparison filter
filter = {
    "column": "age",
    "operator": ">",
    "value": 25
}
df = editor.select("users", filters=filter)
# Generates: SELECT * FROM users WHERE age > ?
# Parameters: [25]
```

### Multiple Conditions with AND

```python
# List of conditions = AND logic (default)
filters = [
    {"column": "age", "operator": ">", "value": 25},
    {"column": "status", "operator": "=", "value": "active"},
    {"column": "city", "operator": "=", "value": "Oslo"}
]
df = editor.select("users", filters=filters)
# Generates: SELECT * FROM users WHERE age > ? AND status = ? AND city = ?
# Parameters: [25, "active", "Oslo"]
```

### Multiple Conditions with OR

```python
# Explicit OR logic
filters = {
    "or": [
        {"column": "role", "operator": "=", "value": "admin"},
        {"column": "role", "operator": "=", "value": "moderator"},
        {"column": "role", "operator": "=", "value": "staff"}
    ]
}
df = editor.select("users", filters=filters)
# Generates: SELECT * FROM users WHERE role = ? OR role = ? OR role = ?
# Parameters: ["admin", "moderator", "staff"]
```

## Supported Operators

### Comparison Operators
- `=` : Equal
- `!=` : Not equal
- `<>` : Not equal (SQL standard)
- `<` : Less than
- `>` : Greater than
- `<=` : Less than or equal
- `>=` : Greater than or equal

```python
filter = {"column": "price", "operator": "<=", "value": 100}
```

### String Matching
- `LIKE` : Pattern matching with % and _ wildcards

```python
filter = {"column": "name", "operator": "LIKE", "value": "%john%"}
```

### List Operations
- `IN` : Match any value in a list
- `NOT IN` : Match no values in a list

```python
filter = {"column": "id", "operator": "IN", "value": [1, 2, 3, 4, 5]}
# Generates: SELECT * FROM users WHERE id IN (?, ?, ?, ?, ?)

filter = {"column": "country", "operator": "NOT IN", "value": ["US", "UK", "CA"]}
# Generates: SELECT * FROM users WHERE country NOT IN (?, ?, ?)
```

### Range Operations
- `BETWEEN` : Value within a range (inclusive)

```python
filter = {"column": "age", "operator": "BETWEEN", "value": [18, 65]}
# Generates: SELECT * FROM users WHERE age BETWEEN ? AND ?
# Parameters: [18, 65]
```

### NULL Operations
- `IS NULL` : Column is null
- `IS NOT NULL` : Column is not null

```python
filter = {"column": "deleted_at", "operator": "IS NULL"}
# Generates: SELECT * FROM users WHERE deleted_at IS NULL
# No parameters needed

filter = {"column": "updated_at", "operator": "IS NOT NULL"}
# Generates: SELECT * FROM users WHERE updated_at IS NOT NULL
```

## Practical Examples

### Example 1: User Search with Multiple Filters

```python
# Find active users aged 25-65 in Oslo
filters = [
    {"column": "age", "operator": ">=", "value": 25},
    {"column": "age", "operator": "<=", "value": 65},
    {"column": "city", "operator": "=", "value": "Oslo"},
    {"column": "status", "operator": "=", "value": "active"}
]
df = editor.select("users", filters=filters, limit=100)
```

### Example 2: Product Search

```python
# Find products in certain categories with price range
filters = [
    {"column": "category", "operator": "IN", "value": ["electronics", "computers"]},
    {"column": "price", "operator": "BETWEEN", "value": [100, 5000]},
    {"column": "stock", "operator": ">", "value": 0}
]
df = editor.select("products", filters=filters)
```

### Example 3: Event Filtering

```python
# Find recent events or upcoming events created by specific users
filters = {
    "or": [
        {
            "and": [
                {"column": "event_date", "operator": ">", "value": "2024-01-01"},
                {"column": "created_by", "operator": "IN", "value": [1, 2, 3]}
            ]
        },
        {"column": "event_type", "operator": "=", "value": "featured"}
    ]
}
# Note: Complex nested structures should be simplified using lists
```

### Example 4: Search with User Input (Safe)

```python
# Building filters from user input - completely safe from injection
def search_users(age_min, age_max, city, name_pattern):
    filters = [
        {"column": "age", "operator": "BETWEEN", "value": [age_min, age_max]},
        {"column": "city", "operator": "=", "value": city},
        {"column": "name", "operator": "LIKE", "value": f"%{name_pattern}%"}
    ]
    return editor.select("users", filters=filters)

# These calls are ALL safe from SQL injection:
search_users(25, 65, "Oslo", "john")
search_users(18, 99, "'; DROP TABLE users; --", "%x%")
search_users(0, 120, "New York", "' OR '1'='1")
```

## Using with count()

The `count()` method also supports structured filters:

```python
# Count active users
count = editor.count("users", 
    filters={"column": "status", "operator": "=", "value": "active"}
)

# Count users in age range
count = editor.count("users",
    filters={"column": "age", "operator": "BETWEEN", "value": [18, 65]}
)

# Count with multiple conditions
count = editor.count("users", filters=[
    {"column": "status", "operator": "=", "value": "active"},
    {"column": "created_at", "operator": ">", "value": "2024-01-01"}
])
```

## Backward Compatibility

The old `where` parameter still works for backward compatibility:

```python
# ⚠️ Old way (still works but deprecated)
df = editor.select("users", where="age > 25 AND status = 'active'")

# ✅ New way (recommended)
df = editor.select("users", filters=[
    {"column": "age", "operator": ">", "value": 25},
    {"column": "status", "operator": "=", "value": "active"}
])
```

**Note**: If both `where` and `filters` are provided, `filters` takes precedence.

## Error Handling

Invalid filter structures raise clear errors:

```python
# Error: Invalid column name
filters = {"column": "user; DROP TABLE;", "operator": "=", "value": 1}
# Raises: SQLInjectionError

# Error: Invalid operator
filters = {"column": "age", "operator": "INVALID", "value": 25}
# Raises: ValueError

# Error: Missing required field
filters = {"column": "age"}  # Missing operator!
# Raises: KeyError or ValueError

# Error: Type mismatch for operator
filters = {"column": "id", "operator": "IN", "value": 123}  # Need list!
# Raises: ValueError
```

## Performance Considerations

- Structured filters are compiled to parameterized SQL, which is just as fast as raw SQL
- DuckDB's query optimizer handles parameterized queries efficiently
- The filtering happens in the database, not in Python

## Security Summary

Structured filters provide multiple layers of security:

1. **Column Name Validation**: Only alphanumeric + underscore allowed
2. **Operator Whitelist**: Only known operators are accepted
3. **Value Parameterization**: All values use DuckDB's parameter binding (`?`)
4. **Type Safety**: Python types prevent most injection attacks at parse time

## Migration Guide

If you're currently using string-based WHERE clauses, here's how to migrate:

### Before (Vulnerable)
```python
def get_users_by_status(status):
    return editor.select("users", where=f"status = '{status}'")

# Risk: status could contain SQL injection
get_users_by_status("active")  # OK
get_users_by_status("active'; DROP TABLE users; --")  # DANGER!
```

### After (Safe)
```python
def get_users_by_status(status):
    return editor.select("users", filters={
        "column": "status",
        "operator": "=",
        "value": status
    })

# Both are safe:
get_users_by_status("active")  # OK
get_users_by_status("active'; DROP TABLE users; --")  # Still safe!
```

## References

- [OWASP: Parameterized Queries](https://cheatsheetseries.owasp.org/cheatsheets/Query_Parameterization_Cheat_Sheet.html)
- [DuckDB Parameter Binding](https://duckdb.org/docs/api/overview)
