#%%
from parquedit import ParquEdit
import pandas as pd
import os
import json
from config.config import settings

# %%
db_config = settings.parquedit.to_dict()
db_config.update({'dbuser': os.getenv("DAPLA_GROUP_CONTEXT")+"@dapla-group-sa-t-57.iam"})
# %%
with ParquEdit(db_config) as editor:
    print(editor.view('vst_table_23', limit=110))
# %%
editor.view?
# %%
