# Databricks notebook source
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# COMMAND ----------

def prep_data(data):
    categorical_cols=data.select_dtypes(include='object')
    numeric_cols=data.select_dtypes(include=np.number)
    data.pdays = data.pdays.replace(999, 0)  
    numeric_cols=data.select_dtypes(include=np.number)
    
    scalar=MinMaxScaler()
    column=numeric_cols.columns
    numeric_cols=scalar.fit_transform(numeric_cols)
    df_numeric=pd.DataFrame(numeric_cols,columns=column)
    
    le=LabelEncoder()
    df_categorical=pd.DataFrame()
    for col in categorical_cols.columns:
        df_categorical[col]=le.fit_transform(categorical_cols[col])
        
    df=pd.concat([df_numeric,df_categorical],axis=1)        
    
    return df

# COMMAND ----------

def get_data(sample_size=3):
    df_data = spark.read.table("default.marketing_campaign").toPandas()
    df_data_sample = df_data.sample(n=sample_size)
    return df_data_sample.iloc[:,0:20], df_data_sample.iloc[:,-1:]

# COMMAND ----------

#Load training data
data = spark.read.table("default.marketing_campaign_train").toPandas()

# COMMAND ----------

df = prep_data(data)
x=df.drop('y',axis=1)
y=df['y']

lr=LogisticRegression(solver='lbfgs', max_iter=3000)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)
model=lr.fit(x_train,y_train)

# COMMAND ----------

predict_test=model.predict(x_test)
matrix = classification_report(y_test,predict_test)
print('Classification report testing data: \n',matrix)

# COMMAND ----------

sample_to_predict, ground_truth = get_data(4)
prepped_for_prediction = prep_data(sample_to_predict)

predicted = model.predict(prepped_for_prediction)

result = ground_truth
result['predicted'] = predicted.tolist()
result

# COMMAND ----------


