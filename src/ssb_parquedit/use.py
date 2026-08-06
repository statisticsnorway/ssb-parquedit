#%%
from ssb_parquedit import ParquEdit
import logging
logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig()

# %%
conn = ParquEdit()
# %%
conn.list_tables()
# %%
conn.get_edits()
# %%
conn.view('test_table_1', where="rowid=999999")
# %%
conn.flush_inlined_table('test_table_1')
# %%
conn.merge_adjacent_files('test_table_1')
# %%
icon = ParquEdit()
raw = conn._get_connection().raw 
# %%
