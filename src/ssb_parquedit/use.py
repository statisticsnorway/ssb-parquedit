# %%
from ssb_parquedit import ParquEdit

# Auto-configure from environment
con = ParquEdit()

# %%
import pandas as pd

df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})

# %%
# Create table from DataFrame (empty — schema only)
con.create_table("inline_table", source=df, product_name="my-product", fill=True)
# %%
con.insert_data("inline_table", source=df)
# %%
con.view("inline_table")
# %%
conn = ParquEdit()
internal_conn = conn._get_connection()._conn
internal_conn.execute("SELECT * FROM inline_table").df()
# %%
internal_conn.execute("CALL ducklake_merge_adjacent_files('dapla_ffunk')")
# %%
for i, _ in enumerate(range(100)):
    con.insert_data("inline_table", source=df)
    print(f"Insert {i+1}/100:")
# %%
internal_conn.execute("SELECT * FROM inline_table").df()

# %%

from ssb_parquedit import ParquEdit

# Auto-configure from environment
conn = ParquEdit()
internal_conn = conn._get_connection().raw
internal_conn = conn._get_connection()._conn
# %%
internal_conn.sql("SELECT * from inline_table")
# %%
