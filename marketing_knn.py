# Databricks notebook source
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,KBinsDiscretizer,Binarizer
from sklearn.compose import make_column_transformer
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# COMMAND ----------

#Load data
data_spark = spark.read.table("default.marketing_campaign_train")
data = data_spark.toPandas()

# COMMAND ----------

# Setup and train
# Remove ground truth (last column)
x = data.iloc[:,0:20]
y = data['y']

# Columntypes
nom_col = [1,2,3,8,9,14]
ordinal_col = [4,5,6,7]
kbins_col = [18]
Bina_col = [0]

trans = make_column_transformer((OneHotEncoder(sparse = False),nom_col),
                                (OrdinalEncoder(),ordinal_col),
                                (KBinsDiscretizer(),kbins_col),
                                (Binarizer(threshold = 55),Bina_col),
                                 remainder = 'passthrough')
set_config(display = 'diagram')

# Get train and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3) 

# Model definition and train
KNN = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
s = SMOTE()
pipeline = make_pipeline(trans,s,KNN)
pipeline.fit(x_train,y_train)

# COMMAND ----------

# Data to predict, random sample from 'production table'
def get_data(sample_size=3):
    df_data = spark.read.table("default.marketing_campaign").toPandas()
    df_data_sample = df_data.sample(n=sample_size)
    return df_data_sample.iloc[:,0:20], df_data_sample.iloc[:,-1:]

# COMMAND ----------

# Get sample and ground truth
sample_to_predict, ground_truth = get_data(4)

# Predict
predicted = pipeline.predict(sample_to_predict)

# COMMAND ----------


