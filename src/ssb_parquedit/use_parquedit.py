# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from parquedit import ParquEdit
import pandas as pd
import os
import json
from config.config import settings

# %%
# henter prosjekt-config
db_config = settings.parquedit.to_dict()
db_config.update({'dbuser': os.getenv("DAPLA_GROUP_CONTEXT")+"@dapla-group-sa-t-57.iam"})


# %%
# Kilde for en tabell kan være en enkel dataframe
old_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'London', 'Tokyo'],
    'year': [2020,2020,2022],
    'month': [1,2,2],
    'week': [1,2,3],
    'day': [10,20,20],
})

# %%
# Kilde kan også være json-schema fra en fil
schema = json.load(open("products.json"))

# %%
# Eller en parquetfil
parquetfile = "gs://ssb-dapla-ffunk-data-hns-test/temp/testdata_1m_rows_50_cols.parquet"

# %%
data.to_parquet(parquetfile)

# %%
# lager en Parquedit-instanse
with ParquEdit(db_config) as editor:
    # oppretter tabeller fra ulike kilder
    #editor.create_table('vst_table_23', old_data, 'very nice table', ['year'], fill=True)    
    #editor.create_table('basic_table_1b', data, 'very nice table', part_columns=[], fill_table=True)    
    #editor.create_table('basic_table_1c', data, 'very nice table', part_columns=[])    
    #editor.create_table('basic_table_2', schema, 'not so nice table')
    #editor.create_table(table_name='unbasic_table_4', source=parquetfile, table_description='worst table ever', fill=True)
    #editor.fill_table('basic_table_1c', data)
    #editor.create_table('basic_table_1d', data, 'very nice table', ['year','age'], fill_table=True)    
    #editor.create_table('basic_table_1e', data, 'very nice table', ['year','age','month', 'week', 'day'], fill_table=True)    
    #editor.fill_table('basic_table_3c', parquetfile)
    #df = editor.select("unbasic_table_1", limit=5)
    #df = editor.select("unbasic_table_1", columns=["age", "name"], filters={"column": "city", "operator": "=", "value": "New York"})
    #df = editor.select("unbasic_table_3", limit=1000, filters=[
    #    {"column": "var_1", "operator": "=", "value": 724},
    #    {"column": "var_2", "operator": "=", "value": 300}
    #], order_by='id')
    #df = editor.select("unbasic_table_3", limit=100000, filters={"column": "var_1", "operator": "=", "value": 724},
    #                            order_by='id', offset=100, columns=["id", "var_10", "var_2"])

                            

    #print(editor.select("unbasic_table_4", limit=100000, filters={"column": "var_1", "operator": "=", "value": 724},
    #                            order_by='id', offset=100, columns=["id", "var_10", "var_2"]))

    print(editor.select("vst_table_23"))

    #print(editor.count("vst_table_23"))


# %%
import time

with ParquEdit(db_config) as editor:
    start = time.perf_counter()
    result = editor.select("unbasic_table_4", 
        filters={"column": "id", "operator": "=", "value": "ffe39a2d-56aa-4dc4-902e-eab2d04ec2cf"})
    end = time.perf_counter()
    
    print(f"Kjøretid: {(end - start)*1000:.2f} ms")
    print(result)

# %%
print(df)
# %%
## Måle tidsbruk på seleksjon av enkeltrad 1 av 10_000_000
import time

with ParquEdit(db_config) as editor:
    start = time.perf_counter()
    result = editor.select("unbasic_table_4", 
        filters={"column": "id", "operator": "=", "value": "ffe39a2d-56aa-4dc4-902e-eab2d04ec2cf"})
    end = time.perf_counter()
    
    print(f"Kjøretid: {(end - start)*1000:.2f} ms")
    print(result)
# %%

new_data2 = pd.DataFrame({
    'name': ['R', 'V', 'J'],
    'age': [23, 32, 37],
    'city': ['New York', 'London', 'Tokyo'],
    'year': [2021,2021,2022],
    'month': [4,2,11],
    'week': [1,2,3],
    'day': [10,20,20],
})


# %%
new_data2.to_parquet("gs://ssb-dapla-ffunk-data-hns-test/temp/new_data.parquet")

#print(new_data)

# %%
parquetfile = "gs://ssb-dapla-ffunk-data-hns-test/temp/freshdata.parquet"

with ParquEdit(db_config) as editor:
    # oppretter tabeller fra ulike kilder
    # opprette tom tabell fra dataframe
    #editor.create_table(table_name="urb_tab_empty",source=new_data2)
    #print(editor.select("urb_tab_empty",limit=40))
    # insert til ny tabell med data fra dataframe
    #editor.insert_data(table_name="urb_tab_empty", source=new_data2)
    #print(editor.select("urb_tab_empty",limit=40))
    # insert til ny tabell med data fra parquet-fil
    #editor.insert_data(table_name="urb_tab_empty", source=parquetfile)    
    #print(editor.select("urb_tab_empty",limit=40))
    # opprette tabell og fylle den
    #editor.create_table(table_name="urb_tab_not_empty",source=new_data2, fill=True)
    # insert til ny tabell med data fra dataframe
    #editor.insert_data(table_name="urb_tab_not_empty", source=new_data2)
    #print(editor.select("urb_tab_empty",limit=40))
    #insert til ny tabell med data fra parquet-fil    
    #editor.insert_data(table_name="urb_tab_not_empty", source=parquetfile)
    #print(editor.select("urb_tab_not_empty",limit=40))
    # ny tabell med partisjon
    editor.create_table(table_name="urb_tab_part",source=new_data2, fill=True, part_columns=["year", "month"])

# %%
