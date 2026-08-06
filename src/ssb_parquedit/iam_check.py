"""Example: wiring iam_auth.py into DuckDBConnection.

This replaces both the rejected Cloud SQL Proxy subprocess approach and the
cloud-sql-python-connector-internals relay with a direct connection. Only
the __init__ method is shown -- the rest of DuckDBConnection is unchanged.
"""
#%%
import duckdb
import gcsfs

from iam_auth import get_instance_ip, get_login_token

#%%

pg_host = "localhost"
pg_password_line = ""
instance_connection_name = "dapla-ffunk-sql-p-xo:europe-north1:parquedit"
        
ip_type = "PRIVATE"
pg_host = get_instance_ip(instance_connection_name, ip_type=ip_type)
token = get_login_token()
# Single-quoted in the DSN below; tokens are base64url and never
# contain a quote, so no escaping is needed.
pg_password_line = f"\n                password={token}\n                sslmode=require"
#%%
print(pg_password_line)
print(token)


class DuckDBConnection:
    _conn: duckdb.DuckDBPyConnection | None = None

    def __init__(self, db_config: dict[str, str]) -> None:
        """Initialize DuckDB connection with DuckLake catalog.

        Args:
            db_config: Database configuration dict with the following keys:

                - ``dbname``: PostgreSQL database name.
                - ``dbuser``: PostgreSQL user. When using IAM auth, this is
                  the IAM principal's email (service accounts: without the
                  ``.gserviceaccount.com`` suffix).
                - ``catalog_name``: Name of the DuckLake catalog to attach.
                - ``data_path``: GCS path for data storage.
                - ``metadata_schema``: PostgreSQL schema for DuckLake metadata.
                - ``cloud_sql_instance_connection_name`` (optional):
                  ``project:region:instance``. If set, the instance's IP is
                  looked up via the Cloud SQL Admin API and a fresh IAM
                  login token is used as the password -- no Proxy, no
                  Connector, no local relay.
                - ``cloud_sql_ip_type`` (optional): ``"PRIVATE"`` (default)
                  or ``"PRIMARY"`` for a public-IP instance.
        """
        self._conn = duckdb.connect()

        fs = gcsfs.GCSFileSystem()
        self._conn.register_filesystem(fs)

        for ext in ("ducklake", "postgres"):
            self._conn.sql(f"INSTALL {ext}")
            self._conn.sql(f"LOAD {ext}")

        pg_host = "localhost"
        pg_password_line = ""
        instance_connection_name = db_config.get("cloud_sql_instance_connection_name")
        if instance_connection_name:
            ip_type = db_config.get("cloud_sql_ip_type", "PRIVATE")
            pg_host = get_instance_ip(instance_connection_name, ip_type=ip_type)
            token = get_login_token()
            # Single-quoted in the DSN below; tokens are base64url and never
            # contain a quote, so no escaping is needed.
            pg_password_line = f"\n                password={token}\n                sslmode=require"

        self._conn.sql(f"""
            ATTACH 'ducklake:postgres:
                dbname={db_config["dbname"]}
                user={db_config["dbuser"]}
                host={pg_host}{pg_password_line}
            ' AS {db_config["catalog_name"]}
            (DATA_PATH '{db_config["data_path"]}',
            METADATA_SCHEMA {db_config["metadata_schema"]},
            DATA_INLINING_ROW_LIMIT 300,
            AUTOMATIC_MIGRATION TRUE);
            """)
        self._conn.sql(f"USE {db_config['catalog_name']}")
# %%
