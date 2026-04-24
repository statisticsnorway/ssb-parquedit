# %%
import logging

logging.getLogger("ssb_parquedit").setLevel(logging.DEBUG)
logging.basicConfig()  # legger til en stdout-handler

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
import time

for i, _ in enumerate(range(100)):
    start = time.perf_counter()
    con.insert_data("inline_table", source=df)
    elapsed = time.perf_counter() - start
    print(f"Insert {i+1}/100: {elapsed:.3f}s")
# %%

import numpy as np

df = pd.DataFrame(
    {
        "name": np.random.choice(
            ["Alice", "Bob", "Charlie", "Diana", "Erik"], size=100
        ),
        "age": np.random.randint(18, 65, size=100),
    }
)
# %%
start = time.perf_counter()
print(con.view("inline_table"))
elapsed = time.perf_counter() - start
print(f"view: {elapsed:.3f}s")
# %%
start = time.perf_counter()
print(internal_conn.execute("SELECT * FROM inline_table").df())
elapsed = time.perf_counter() - start
print(f"view: {elapsed:.3f}s")
# %%
con.close()
# %%
