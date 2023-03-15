# Databricks notebook source
import pandas as pd

df_from_table = spark.read.table("default.marketing_campaign_train")
df_from_table.head(3)

# COMMAND ----------

df_from_table.columns

# COMMAND ----------

X = df_from_table.columns[:-1]
y = df_from_table.columns[-1]

# COMMAND ----------


