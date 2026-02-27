# %%
import json
import os

from ssb_parquedit.config.config import settings
from ssb_parquedit.parquedit import ParquEdit

# %%
db_config = settings.parquedit.to_dict()
db_config.update(
    {"dbuser": os.getenv("DAPLA_GROUP_CONTEXT") + "@dapla-group-sa-t-57.iam"}
)
# %%
with ParquEdit(db_config) as editor:
    print(editor.view("vst_table_23", limit=110))

# %%

# Eller en parquetfil
parquetfile = "gs://ssb-dapla-ffunk-data-hns-test/temp/freshdata.parquet"
schema = json.load(open("products.json"))

with ParquEdit(db_config) as editor:
    # editor.create_table('vst_table_27', parquetfile, part_columns=['year'], fill=True)
    # print(editor.view('vst_table_27', limit=110,output_format='pyarrow').read_all())
    # print(editor.view('vst_table_27', limit=110,output_format='polars'))
    # editor.insert_data('vst_table_27', parquetfile)
    # print(editor.count('vst_table_27'))
    # print(editor.exists('vst_table_27'))
    editor.create_table("vst_table_28", schema, part_columns=["name"], fill=False)

# %%
