#%%
from ssb_parquedit.parquedit import ParquEdit
from functions import get_dapla_group
from functions import get_team_name
from functions import get_bucket_name

import duckdb

# %%

short_name= "ffunk_ducklake"

db_config: dict[str, str] = {
    "short_name": "ffunk_ducklake",
    "dbname": "dapla-ffunk",
    "dbuser": f"{get_dapla_group()}@dapla-group-sa-t-57.iam",
    "data_path": f"gs://{get_bucket_name()}/{short_name}/.parquedit_data",
    "catalog_name": get_team_name().replace("-", "_"),
    "metadata_schema": get_team_name().replace("-", "_"),
}


print(db_config)
#%%
parquet_file="gs://ssb-dapla-ffunk-data-hns-test/temp/freshdata.parquet"

pe = ParquEdit(db_config)

pe.create_table(
    table_name="vst_table_2",
    source=parquet_file,
    short_name="ffunk_short_name",
    fill=True
)
#%%
pe.view(
    table_name="vst_table_2",
    )
# %%
pe.count(
    table_name="vst_table_2",
    )
# %%
pe.insert_data(table_name="vst_table_2", source=parquet_file)
# %%
