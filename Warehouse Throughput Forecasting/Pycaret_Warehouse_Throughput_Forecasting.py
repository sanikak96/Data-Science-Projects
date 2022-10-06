# Databricks notebook source
# MAGIC %md
# MAGIC #### Warehouse Throughput Forecasting using Pycaret (Regression)
# MAGIC **What is PyCaret?**
# MAGIC - It is an open-source low-code machine learning library in Python that aims to reduce the time needed for experimenting with different machine learning models.
# MAGIC - It helps Data Scientist to perform any experiments end-to-end quickly and more efficiently and focus on business problems.
# MAGIC - PyCaret is a wrapper around many ML models and frameworks such as XGBoost, Scikit-learn, and many more.
# MAGIC 
# MAGIC **Installing Pycaret**
# MAGIC - %pip install pycaret (In Databricks)
# MAGIC 
# MAGIC 
# MAGIC **[Reference Link 1](https://towardsdatascience.com/introduction-to-regression-in-python-with-pycaret-d6150b540fc4)**
# MAGIC 
# MAGIC **[Reference Link 2](https://pycaret.gitbook.io/docs/learn-pycaret/official-blog/time-series-101-for-beginners)**

# COMMAND ----------

# MAGIC %pip install pycaret

# COMMAND ----------

!pip install jupyter_black

# COMMAND ----------

# Importing Necessary Libraries for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_pacf

#Importing pycaret - Machine Learning (Regression) Library
from pycaret.regression import *

#Follow pep8 standards
import jupyter_black
jupyter_black.load()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading the data

# COMMAND ----------

file_location = "abfss://ml-suite@fiepmqasa.dfs.core.windows.net/HCE-Dublin/WTF/warehouse_hourly_data.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
warehouse_data = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# COMMAND ----------

# converting to pandas df
warehouse_data = warehouse_data.toPandas()

# print information about the data
warehouse_data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating a lag of picking efficiency
# MAGIC We will be forecasting picking efficiency for the next 1 hour, as per ML XO5 definition. Hence, we need to have the target variable as 'Picking_Efficiency_Next_Hour'. Also, from the graph below, we can see that picking efficiency for the current hour is highly correlated to the picking efficiency 1 hour ago.

# COMMAND ----------

plt.rc("figure", figsize=(5, 3))
plot_pacf(warehouse_data["Picking_Efficiency"], lags=30)
plt.tight_layout()
plt.show()

# COMMAND ----------

warehouse_data_new = warehouse_data.copy()
warehouse_data_new['Picking_Efficiency_Next_Hour'] = warehouse_data['Picking_Efficiency'].shift(-1)
data_unseen = warehouse_data_new[-1:]
warehouse_data_new = warehouse_data_new[:-1]
data_unseen

# COMMAND ----------

data_unseen.drop('Picking_Efficiency_Next_Hour',axis=1, inplace=True)
data_unseen

# COMMAND ----------

# MAGIC %md
# MAGIC #### Splitting the data into train and test sets

# COMMAND ----------

warehouse_data_new = warehouse_data_new.drop(columns = ['Day', 'Month', 'Year', 'Day of the week', 'Weekday'], axis=1)

# COMMAND ----------

warehouse_data_new["Date"] = pd.to_datetime(
    warehouse_data_new.Date, infer_datetime_format="True"
)
split_date = pd.datetime(2020, 11, 29)
train = warehouse_data_new.loc[warehouse_data_new.Date < split_date]
test = warehouse_data_new.loc[warehouse_data_new.Date >= split_date]

test.shape

# COMMAND ----------

test

# COMMAND ----------

print('Data for Modeling: ' + str(train.shape))
print('Unseen Data For Predictions: ' + str(test.shape))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Setting up PyCaret Regression Module for Time Series Forecasting
# MAGIC - PyCaret’s Regression module default settings are not ideal for time series data because it involves few data preparatory steps that are not valid for ordered data. 
# MAGIC - For example, the split of the dataset into train and test set is done randomly with shuffling. This wouldn’t make sense for time series data. 
# MAGIC - So, I have to specified the train and test data explicitly after splitting them and have used the timeseries fold strategy pre-built in pycaret.

# COMMAND ----------

regression_model = setup(data = train, test_data = test, target = 'Picking_Efficiency_Next_Hour', fold_strategy = 'timeseries', session_id=1, silent = True, html = False, data_split_shuffle = False) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Comparing models
# MAGIC Models for regression have been evaluated using 10 fold cross validation and the most commonly use regression metrics.

# COMMAND ----------

compare_models(sort = 'mse')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Creating the best model as given by pycaret

# COMMAND ----------

model1 = create_model('omp')
model1

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Tuning the model

# COMMAND ----------

tuned_model = tune_model(model1,n_iter = 50)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Predictions

# COMMAND ----------

predictions = predict_model(tuned_model)
predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Printing the bounds of picking efficiency for our dataset
# MAGIC 
# MAGIC - Since picking efficiency = volume of units processed / Labour time, the range can vary depending on these two features. 
# MAGIC - For example, if warehouse A processes 400 units in 20 mins of labour work, the picking efficiency will be 20; where as if warehouse B processes only 10 goods in the same labour time, picking efficiency will be 0.5. 
# MAGIC - This in no way implies that 0.5 is worse than 20 since number of goods processed hugely depends on the type of warehouse and the type of goods. 
# MAGIC - Thus, for better understanding of what the output means, bounds of picking efficiency of this data has been printed.
# MAGIC 
# MAGIC We can see that for the entire data, the least picking efficiency value is 1.68 and the best is 6.198.

# COMMAND ----------

print('Min and Max Values of picking efficiency in the whole data are: ',round(warehouse_data['Picking_Efficiency'].min(),3), 'and' ,round(warehouse_data['Picking_Efficiency'].max(),3))
print('Min and Max Values of picking efficiency in test data are: ',round(predictions['Picking_Efficiency'].min(),3), 'and' ,round(predictions['Picking_Efficiency'].max(),3))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Plotting the predictions
# MAGIC - Y-axis denotes the picking efficiency based on the bounds mentioned above.
# MAGIC - X-axis shows the hour of the day.
# MAGIC - The Red line shows the values predicted by the model and blue line shows the actual values as per the dataset.

# COMMAND ----------

predictions['Date'] = pd.date_range(start='2022-11-29', periods=24, freq = 'H')
predictions = predictions.rename(columns={"Label": "Predicted_Picking_Efficiency_Next_Hour"})
# line plot
fig = px.line(predictions, x='Date', y=["Picking_Efficiency_Next_Hour", "Predicted_Picking_Efficiency_Next_Hour"], template = 'plotly_dark',
              labels={'Date':'Hour','value':'Picking Efficiency'} 
)
fig.update(layout_yaxis_range = [1.68 , 6.198])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### A note on data leakage
# MAGIC - Data leakage is the use of information during model training which would not be available during prediction stage.
# MAGIC - It can be due to the following reasons:
# MAGIC   - Your training data includes information about your test set
# MAGIC   - The feature is simply not available in the future, or, the feature can be made available but it is expensive to forecast it and come with lots of uncertainties/errors
# MAGIC   
# MAGIC To avoid data leakage:
# MAGIC   - For timeseries problem, train-test split has been done manually and without shuffling. This ensures that future data is not seen. 
# MAGIC   - For this PoC, Picking efficiency of the next hour is calculated using the features of previous hour. Thus, forecasting them doesn't come into picture.
# MAGIC   
# MAGIC Future work can be done to identify significant lags of each feature that contribute to the picking efficiency based on properties like seasonality.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving the final model

# COMMAND ----------

save_model(tuned_model,'Throughput Model pycaret')

# COMMAND ----------

saved_final_model = load_model('Throughput Model pycaret')
saved_final_model

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Checking if the model is saved

# COMMAND ----------

dbutils.fs.cp( '/home/spark-d4d8d5b4-82be-43dc-a6d0-c2/Throughput Model pycaret.pkl', 'abfss://ml-suite@fiepmqasa.dfs.core.windows.net/HCE-Dublin/WTF/')

# COMMAND ----------

dbutils.fs.ls('abfss://ml-suite@fiepmqasa.dfs.core.windows.net/HCE-Dublin/WTF/')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6. Prediction for Unseen Data
# MAGIC In this dataset we have data till 30th November 12 am. On feeding the data of 30th Nov 12 am, we can predict throughput for the next 1 hour. 
# MAGIC 
# MAGIC The model has predicted picking efficiency for 1am.

# COMMAND ----------

data_unseen

# COMMAND ----------

prediction_unseen = predict_model(saved_final_model, data = data_unseen)
prediction_unseen = prediction_unseen.rename(columns={"Label": "Predicted_Picking_Efficiency_Next_Hour"})
prediction_unseen

# COMMAND ----------

# MAGIC %md
# MAGIC #### Conclusion
# MAGIC - The pycaret modules can be used for quick analysis of many regression algorithms and see how they are working.
# MAGIC - The model works good and can predict the picking efficiency for next one hour.
