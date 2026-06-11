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

A Python package for manually editing tabular data stored as Parquet files on [DaplaLab](https://manual.dapla.ssb.no/) — Statistics Norway's cloud data platform. Built on top of [DuckDB](https://duckdb.org/) and the [DuckLake](https://ducklake.select/) catalog, it provides a clean Python interface for creating tables, inserting data, querying results and editing rows directly from Google Cloud Storage (GCS).
Intended for single-table editing. Does not support primary- and foreign keys.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic setup](#basic-setup)
  - [Creating a table](#creating-a-table)
  - [Inserting data](#inserting-data-in-an-existing-table)
  - [Editing a row](#editing-a-row)
  - [Querying data](#querying-data)
  - [Counting rows](#counting-rows)
  - [Checking table existence](#checking-table-existence)
  - [List all tables](#list-all-tables)
  - [List edits](#list-edits)
  - [Drop table](#drop-table)
- [Advanced](#advanced)
  - [Accessing the raw DuckDB connection](#accessing-the-raw-duckdb-connection)
  - [Setting up local connection](#setting-up-local-connection)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Auto-configuration** — reads Dapla environment variables to build connection config automatically
- **DuckLake catalog integration** — metadata stored in PostgreSQL, data stored in GCS
- **Create tables** from a pandas DataFrame, a JSON Schema dict, or an existing GCS Parquet file
- **Insert data** from a pandas DataFrame or a `gs://` Parquet path — rows are automatically assigned a unique `rowid` within a table
- **Edit data** - Update value(s) in a single row by its rowid.
- **Query tables** with where-conditions, column selection, sorting, pagination, and multiple output formats (`pandas`, `polars`, `pyarrow`)
- **Find edits** Retrieve historical column-level edits for a specified table
- **Count rows**
- **Check table existence** safely
- **Partition tables** by one or more columns


---

## Requirements

- Python `>=3.12`
- Access to a DaplaLab environment
- A PostgreSQL instance reachable at `localhost` for DuckLake metadata storage
- A GCS bucket following the naming convention `ssb-{team-name}-data-produkt-{environment}`

### Python dependencies

| Package    | Version              |
|------------|----------------------|
| `duckdb`   | `==1.5.2`            |
| `pandas`   | `>=3.0.0, <4.0.0`   |
| `polars`   | `>=1.38.1, <2.0.0`  |
| `pyarrow`  | `>=23.0.1, <24.0.0` |
| `gcsfs`    | `>=2026.1.0, <2027.0.0` |
| `click`    | `>=8.0.1`            |
| `tenacity` | `>=9.1.4,<10.0.0`    |

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

# Option 1: Create from DataFrame (empty — schema only)
con.create_table(table_name="my_table_1",
                 source=df,
                 product_name="my-product",
                 user_defined_id=["name"])
```
```python

# Option 2: Create and immediately populate with data
con.create_table(table_name="my_table_2",
                 source=df,
                 product_name="my-product",
                 user_defined_id=["name"],
                 fill=True)
```
```python

# Option 3: Create from a JSON Schema
schema = {
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"},
    }
}
con.create_table(table_name="my_table_3",
                 source=schema,
                 product_name="my-product",
                 user_defined_id=["name"])
```
```python

# Option 4: Create from an existing GCS Parquet file (schema inferred from file)
con.create_table(table_name="my_table_4",
                 source="gs://my-bucket/path/to/file.parquet",
                 product_name="my-product",
                 user_defined_id=["id", "year"])
```
```python

# Option 5: Create with partitioning and immediately populate with data
con.create_table(table_name="my_table_5",
                 source=df,
                 product_name="my-product",
                 part_columns=["age"],
                 user_defined_id=["name"],
                 fill=True)
```

> **Notes:**
> - `product_name` is required and is stored as a comment on the table.
> - `table_name` must be lowercase, start with a letter or underscore, contain only lowercase letters, numbers, and underscores, and be at most 20 characters.
> - `user_defined_id` — a list of columns that together uniquely identify a row in a table, used to mimic a primary key.

### Inserting data in an existing table
```python
# Insert from a DataFrame
con.insert_data(table_name="my_table_1",
                 source=df)
```
```python
# Insert from a GCS Parquet file
con.insert_data(table_name="my_table_4",
                 source="gs://my-bucket/path/to/file.parquet")
```
Each inserted row is automatically assigned a unique `rowid` within the table


### Editing a row
`edit()` updates exactly one row — identified by its `rowid` — and logs the change reason and comment to the DuckLake snapshot.
```python
# First look up the rowid of the row you want to edit
result = con.view(table_name="my_table_1",
                  where="name = 'Alice'")
rowid = result["rowid"].iloc[0]

# Then edit it
con.edit(
    table_name="my_table_1",
    rowid=rowid,
    changes={"name":"Alice B", "age": 33},
    change_event_reason="REVIEW",
    change_comment="Corrected name and age after data review",
)
```
`changes` is a dict of `{column_name: new_value}` pairs.

`change_event_reason` must be one of: `OTHER_SOURCE`, `REVIEW`, `OWNER`, `MARGINAL_UNIT`, `DUPLICATE`, `OTHER`


### Querying data
```python
# View all rows (returns pandas DataFrame by default)
result = con.view(table_name="my_table_1")
```
```python
# Filter with a WHERE clause
result = con.view(table_name="my_table_1", where="age > 25")
result = con.view(table_name="my_table_1", where="name = 'Alice' AND age >= 30")
```
```python
# Limit and offset (pagination)
result = con.view(table_name="my_table_1",
                  limit=10,
                  offset=2)
```
```python
# Select specific columns
result = con.view(table_name="my_table_1",
                  columns=["name", "age"])
```
```python
# Sort results
result = con.view(table_name="my_table_1",
                   order_by="age DESC")
```
```python
# Return as polars or pyarrow
result = con.view(table_name="my_table_1",
                   output_format="polars")

result = con.view(table_name="my_table_1",
                   output_format="pyarrow")
```

### Counting rows
```python
total = con.count(table_name="my_table_1",
                   where="name='Alice'")
```

### Checking table existence
```python
if con.exists(table_name="my_table_1"):
    print("Table found")
```

### List all tables
```python
con.list_tables()
```

### List edits
`get_edits()` - Retrieves the full changelog for a table by reading DuckLake snapshot metadata.
Each row represents a single edit, with columns for who made the change, when,
the reason, which row was affected (identified by its unique key), and the old
and new values for all modified columns.

Optionally filter by table name, or omit it to get the changelog for all tables.
```python
# All edits for a specific table
df = con.get_edits(table_name="my_table")

# All edits across all tables
df = con.get_edits()
```

The returned DataFrame includes these changelog columns:

| Column | Description |
|---|---|
| `changed_by` | User who made the edit |
| `change_event_reason` | Reason code (e.g. `REVIEW`, `OWNER`) |
| `change_comment` | Free-text comment from the editor |
| `table_name` | Table the edit was made on |
| `rowid` | Internal row identifier |
| `user_defined_id` | Business key values identifying the row |
| `old_values` | Dict of column → old value for changed columns |
| `new_values` | Dict of column → new value for changed columns |
| `product_name` | Product name the table belongs to |

### Drop table
`drop_table()` - Drops a table from the DuckLake catalog. By default, only removes the table from the catalog. DuckLake preserves data files and snapshot history, so edit history remains accessible via get_edits() after a normal drop.
When purge=True, additionally expires snapshots and deletes GCS data files. This permanently destroys all history and cannot be undone.

```python
# Removes the table from the catalog
con.drop_table(table_name="my_table")
```
```python
# Removes the table from the catalog, expires snapshots and deletes data files
con.drop_table(table_name="my_table", purge=True)
```

---

## Advanced

### Accessing the raw DuckDB connection

`ParquEdit` wraps a `DuckDBConnection`, which exposes the underlying `duckdb.DuckDBPyConnection` via its `.raw` property. This is useful when integrating with libraries that require a native DuckDB connection, such as [Ibis](https://ibis-project.org/).

```python
import ibis
from ssb_parquedit import ParquEdit

con = ParquEdit()
raw = con._get_connection().raw  # duckdb.DuckDBPyConnection

ibis_conn = ibis.duckdb.connect(conn=raw)
table = ibis_conn.table("my_table_1")
```

> **Notes:**
> - `_get_connection()` is an internal method. The raw connection shares state with `ParquEdit` — closing either will affect both. Do not close the raw connection manually while `ParquEdit` is still in use.
> - When using the raw connection, the user is resposible to provide the required information that `ParquEdit`-methods gives. E.g when creating and editing tables.


### Setting up local connection
Create a ParquEdit instance backed by a persistent local SQLite catalog. Useful for local development and testing without GCS or PostgreSQL access. The catalog and data files are stored at ``path`` and persist across sessions. The directory is created if it does not already exist.
```python
con = ParquEdit().local(path="/home/onyxia/work/")
```

---

## Project structure
```text
src/ssb_parquedit/
├── parquedit.py      # ParquEdit facade — main public API
├── connection.py     # DuckDB + DuckLake catalog connection management
├── ddl.py            # DDL operations (CREATE TABLE, partitioning)
├── dml.py            # DML operations (INSERT, EDIT)
├── query.py          # Query operations (SELECT, COUNT, EXISTS)
├── functions.py      # Environment helpers (Dapla config auto-detection)
├── local.py          # Local DuckDB connection backed by SQLite (dev/testing)
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
