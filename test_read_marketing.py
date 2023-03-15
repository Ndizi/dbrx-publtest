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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_from_table[X], df_from_table[y], 
                                                    stratify = df_from_table[y], 
                                                    shuffle = True, 
                                                    test_size = 0.2,
                                                    random_state=2023)

# COMMAND ----------


