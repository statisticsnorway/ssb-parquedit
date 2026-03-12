# %%
from ssb_parquedit.parquedit import ParquEdit

pe = ParquEdit()


# %%
parquet_file = "gs://ssb-dapla-ffunk-data-hns-test/temp/freshdata.parquet"

pe.create_table(
    table_name="vst_table_11",
    source=parquet_file,
    product_name="product_name_ffunk",
    fill=True,
)
# %%
pe.view(
    table_name="vst_table_10",
)
# %%
pe.count(
    table_name="vst_table_10",
)
# %%
pe.insert_data(table_name="vst_table_3", source=parquet_file)
# %%
pe.view(table_name="vst_table_10", output_format="pandas")
# %%
parquet_file = (
    "gs://ssb-dapla-ffunk-data-hns-test/temp/testdata_1m_rows_50_cols.parquet"
)
pe.create_table(
    table_name="big_data_1",
    source=parquet_file,
    product_name="ffunk_short_name",
    fill=True,
)
# %%
pe.view(
    table_name="big_data_1",
    output_format="pandas",
    filters={"column": "var_1", "operator": "=", "value": 579},
)
# %%
