#%%
from ssb_parquedit.parquedit import ParquEdit

# %%
short_name= "ffunk_ducklake"
db_config = ParquEdit.create_config(short_name)
pe = ParquEdit(db_config)
print(db_config)

#%%
parquet_file="gs://ssb-dapla-ffunk-data-hns-test/temp/freshdata.parquet"

pe.create_table(
    table_name="Vst_table_6",
    source=parquet_file,
    short_name="testingtesting",
    fill=True
)
#%%
pe.view(
    table_name="vst_table_3",
    )
# %%
pe.count(
    table_name="vst_table_2",
    )
# %%
pe.insert_data(table_name="vst_table_3", source=parquet_file)
# %%
pe.view(
    table_name="vst_table_3",
    output_format="pyarrow"
    )
# %%
parquet_file="gs://ssb-dapla-ffunk-data-hns-test/temp/testdata_1m_rows_50_cols.parquet"
pe.create_table(
    table_name="big_data_1",
    source=parquet_file,
    short_name="ffunk_short_name",
    fill=True
)
# %%
pe.view(
    table_name="big_data_1",
    output_format="pandas",
    filters={"column": "var_1", "operator": "=", "value": 579}
    )
# %%
