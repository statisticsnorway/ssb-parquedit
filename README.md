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

## Features

- TODO TO DONT

## Requirements

- TODO

## Installation

You can install _SSB Parquedit_ via [pip] from [PyPI]:

```console
pip install ssb-parquedit
```

## Flowchart
```mermaid
    flowchart TD
        User(["👤 User"])

        User -->|"pe = ParquEdit(db_config)"| ParquEdit

        subgraph ParquEdit["parquedit.py — ParquEdit"]
            create_table["create_table()"]
            insert_data["insert_data()"]
            view["view()"]
            count["count()"]
            exists["exists()"]
        end

        subgraph Connection["connection.py — DuckDBConnection"]
            init["__init__()
            - Register GCS filesystem
            - Install/load extensions
            - ATTACH DuckLake catalog
            - USE catalog"]
            enter["__enter__()"]
            exit["__exit__() → close()"]
        end

        subgraph DDL["ddl.py — DDLOperations"]
            ddl_create["create_table()
            - from DataFrame
            - from Parquet
            - from schema dict"]
        end

        subgraph DML["dml.py — DMLOperations"]
            dml_insert["insert_data()
            - from DataFrame
            - from Parquet"]
        end

        subgraph Query["query.py — QueryOperations"]
            q_view["view()"]
            q_count["count()"]
            q_exists["table_exists()"]
        end

        subgraph Functions["functions.py"]
            get_dapla_group["get_dapla_group()
            ← DAPLA_GROUP_CONTEXT"]
            get_team_name["get_team_name()
            e.g. dapla-ffunk"]
            get_bucket_name["get_bucket_name()
            ← DAPLA_ENVIRONMENT"]
        end

        subgraph Config["db_config dict"]
            cfg["dbname
            dbuser
            catalog_name
            data_path
            metadata_schema"]
        end

        Functions --> Config
        Config --> ParquEdit

        create_table -->|"with _get_connection()"| Connection
        insert_data -->|"with _get_connection()"| Connection
        view -->|"with _get_connection()"| Connection
        count -->|"with _get_connection()"| Connection
        exists -->|"with _get_connection()"| Connection

        Connection --> DDL
        Connection --> DML
        Connection --> Query

        subgraph Storage["External"]
            GCS["☁️ GCS Bucket"]
            Postgres["🐘 PostgreSQL
            (DuckLake metadata)"]
        end

        Connection -->|"gcsfs"| GCS
        Connection -->|"ducklake + postgres ext"| Postgres
```

## Usage

Please see the [Reference Guide] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_SSB Parquedit_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [Statistics Norway]'s [SSB PyPI Template].

[statistics norway]: https://www.ssb.no/en
[pypi]: https://pypi.org/
[ssb pypi template]: https://github.com/statisticsnorway/ssb-pypitemplate
[file an issue]: https://github.com/statisticsnorway/ssb-parquedit/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/statisticsnorway/ssb-parquedit/blob/main/LICENSE
[contributor guide]: https://github.com/statisticsnorway/ssb-parquedit/blob/main/CONTRIBUTING.md
[reference guide]: https://statisticsnorway.github.io/ssb-parquedit/reference.html
