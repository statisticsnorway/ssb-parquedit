# SSB Parquedit

[![PyPI](https://img.shields.io/pypi/v/ssb-parquedit.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/ssb-parquedit.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/ssb-parquedit)][pypi status]
[![License](https://img.shields.io/pypi/l/ssb-parquedit)][license]

[![Documentation](https://github.com/statisticsnorway/ssb-parquedit/actions/workflows/docs.yml/badge.svg)][documentation]
[![Tests](https://github.com/statisticsnorway/ssb-parquedit/actions/workflows/tests.yml/badge.svg)][tests]
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_ssb-parquedit&metric=coverage)][sonarcov]
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_ssb-parquedit&metric=alert_status)][sonarquality]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)][poetry]

[pypi status]: https://pypi.org/project/ssb-parquedit/
[documentation]: https://statisticsnorway.github.io/ssb-parquedit
[tests]: https://github.com/statisticsnorway/ssb-parquedit/actions?workflow=Tests
[sonarcov]: https://sonarcloud.io/summary/overall?id=statisticsnorway_ssb-parquedit
[sonarquality]: https://sonarcloud.io/summary/overall?id=statisticsnorway_ssb-parquedit
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[poetry]: https://python-poetry.org/

A Python package for manually editing tabular data stored as Parquet files on [DaplaLab](https://manual.dapla.ssb.no/) — Statistics Norway's cloud data platform. Built on top of [DuckDB](https://duckdb.org/) and the [DuckLake](https://ducklake.select/) catalog, it provides a clean, SQL-injection-safe Python interface for creating tables, inserting data, and querying results directly from Google Cloud Storage (GCS).
Intended use on single-table editing. Does not support primary- and foreign keys.

---

## Features

- **Auto-configuration** — reads Dapla environment variables to build connection config automatically
- **DuckLake catalog integration** — metadata stored in PostgreSQL, data stored in GCS
- **Create tables** from a pandas DataFrame, a JSON Schema dict, or an existing GCS Parquet file
- **Insert data** from a pandas DataFrame or a `gs://` Parquet path — rows are automatically assigned a unique `_id` (UUID)
- **Query tables** with structured filters, column selection, sorting, pagination, and multiple output formats (`pandas`, `polars`, `pyarrow`)
- **Count rows** with optional structured filter conditions
- **Check table existence** safely
- **Partition tables** by one or more columns
- **SQL injection prevention** — all user-supplied filter values are parameterized; column names, table names, and `ORDER BY` clauses are validated against strict allowlists


---

## Requirements

- Python `>=3.12`
- Access to a DaplaLab environment
- A PostgreSQL instance reachable at `localhost` for DuckLake metadata storage
- A GCS bucket following the naming convention `ssb-{team-name}-data-produkt-{environment}`

### Python dependencies

| Package    | Version              |
|------------|----------------------|
| `duckdb`   | `==1.5.1`            |
| `pandas`   | `>=3.0.0, <4.0.0`   |
| `polars`   | `>=1.38.1, <2.0.0`  |
| `pyarrow`  | `>=23.0.1, <24.0.0` |
| `gcsfs`    | `>=2026.1.0, <2027.0.0` |
| `click`    | `>=8.0.1`            |

---

## Installation
```console
poetry add ssb-parquedit
```

---

## Usage

### Basic setup

`ParquEdit` reads its connection configuration automatically from Dapla-environment variables.
```python
from ssb_parquedit import ParquEdit

# Auto-configure from environment
con = ParquEdit()
```

### Creating a table

Tables can be created from a DataFrame schema, a JSON Schema dict, or an existing Parquet file.
```python
import pandas as pd

df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})

# Create table from DataFrame (empty — schema only)
con.create_table("my_table_1",
                 source=df,
                 product_name="my-product")
```

```python
# Create and immediately populate with data
con.create_table("my_table_2",
                 source=df,
                 product_name="my-product",
                 fill=True)
```

```python
# Create from a JSON Schema
schema = {
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"},
    }
}
con.create_table("my_table_3",
                 source=schema,
                 product_name="my-product")
```

```python
# Create from an existing GCS Parquet file (schema inferred from file)
con.create_table("my_table_4",
                 source="gs://my-bucket/path/to/file.parquet",
                 product_name="my-product")

```

```python
# Create with partitioning and immediately populate with data
con.create_table("my_table_5",
                 source=df,
                 product_name="my-product",
                 part_columns=["age"],
                 fill=True)
```

> **Note:** `product_name` is required and is stored as a comment on the table. Table names must be lowercase, start with a letter or underscore, and contain only lowercase letters, numbers, and underscores (max 20 characters).


### Inserting data in an existing table
```python
# Insert from a DataFrame
con.insert_data("my_table_1",
                 source=df)

# Insert from a GCS Parquet file
con.insert_data("my_table_4",
                 source="gs://my-bucket/path/to/file.parquet")
```
Each inserted row is automatically assigned a unique `_id` (UUID string).

### Querying data
```python
# View all rows (returns pandas DataFrame by default)
result = con.view("my_table_1")
```

```python
# Limit and offset (pagination)
result = con.view("my_table_1",
                  limit=10,
                  offset=2)
```
```python
# Select specific columns
result = con.view("my_table_1",
                  columns=["name", "age"])
```
```python
# Sort results
result = con.view("my_table_1",
                   order_by="age DESC")
```
```python
# Return as polars or pyarrow
result = con.view("my_table_1",
                   output_format="polars")

result = con.view("my_table_1",
                   output_format="pyarrow")
```

### Filtering

Filters are structured dicts — **never raw SQL strings** — ensuring SQL injection safety.
```python
# Single condition
con.view("my_table_1",
         filters={"column": "age", "operator": ">", "value": 25})
```

```python
# Multiple conditions (implicit AND)
con.view("my_table_1",
        filters=[
            {"column": "age", "operator": ">", "value": 25},
            {"column": "name", "operator": "LIKE", "value": "A%"},
        ])
```
```python
# Explicit AND / OR
con.view("my_table_1",
        filters={
            "or": [
                {"column": "name", "operator": "=", "value": "Alice"},
                {"column": "name", "operator": "=", "value": "Bob"},
            ]
        })
```
```python
# IN operator
con.view("my_table_1",
          filters={"column": "age", "operator": "IN", "value": [25, 30, 35]})
```
```python
# BETWEEN operator
con.view("my_table_1",
          filters={"column": "age", "operator": "BETWEEN", "value": [20, 40]})
```
```python
# NULL checks
con.view("my_table_1",
          filters={"column": "name", "operator": "IS NOT NULL"})
```

Supported operators: `=`, `!=`, `<>`, `<`, `>`, `<=`, `>=`, `LIKE`, `IN`, `NOT IN`, `BETWEEN`, `IS NULL`, `IS NOT NULL`.

### Counting rows
```python
total = con.count("my_table_1")
active_adults = con.count("my_table_1",
                            filters=[
                                {"column": "age", "operator": ">=", "value": 18},
                            ])
```

### Checking table existence
```python
if con.exists("my_table_1"):
    print("Table found")
```

### List all tables
```python
con.list_tables()
```

---

## Security

SSB Parquedit is designed with SQL injection prevention as a first-class concern.

Key points:
- All filter **values** are passed as parameterized query parameters (never interpolated into SQL strings)
- **Column names**, **table names**, and **ORDER BY** clauses are validated against strict allowlists before being used in query construction
- Raw SQL string filters are not accepted

---
## Project structure
```text
src/ssb_parquedit/
├── parquedit.py      # ParquEdit facade — main public API
├── connection.py     # DuckDB + DuckLake catalog connection management
├── ddl.py            # DDL operations (CREATE TABLE, partitioning)
├── dml.py            # DML operations (INSERT)
├── query.py          # Query operations (SELECT, COUNT, EXISTS)
├── functions.py      # Environment helpers (Dapla config auto-detection)
└── utils.py          # Schema utilities and SQL sanitization
```
---

## Contributing

Contributions are very welcome. To learn more, see the [Contributor Guide].

---

## License

Distributed under the terms of the [MIT license][license]. SSB Parquedit is free and open source software.

---

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

---

## Credits

This project was generated from [Statistics Norway]'s [SSB PyPI Template]. Maintained by Team Fellesfunksjoner at Statistics Norway (Data Enablement Department 724).

[statistics norway]: https://www.ssb.no/en
[pypi]: https://pypi.org/
[ssb pypi template]: https://github.com/statisticsnorway/ssb-pypitemplate
[file an issue]: https://github.com/statisticsnorway/ssb-parquedit/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/statisticsnorway/ssb-parquedit/blob/main/LICENSE
[contributor guide]: https://github.com/statisticsnorway/ssb-parquedit/blob/main/CONTRIBUTING.md
[reference guide]: https://statisticsnorway.github.io/ssb-parquedit/reference.html
