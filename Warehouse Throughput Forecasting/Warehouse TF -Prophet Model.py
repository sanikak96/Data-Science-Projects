# Databricks notebook source
# MAGIC %md
# MAGIC # Warehouse Throughput Forecasting - Using Prophet

# COMMAND ----------

# MAGIC %md
# MAGIC ### About Prophet
# MAGIC - Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
# MAGIC - Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.
# MAGIC - It takes two main input: ds(Date) and y(Target Variable).
# MAGIC - Prophet works well for multivariate Time series forecasting.
# MAGIC 
# MAGIC **Advantages of Prophet**:
# MAGIC - Accurate and fast.
# MAGIC - Prophet is robust to outliers, missing data, and dramatic changes in your time series.
# MAGIC - The Prophet procedure includes many possibilities for users to tweak and adjust forecasts. You can use human-interpretable parameters to improve your forecast by adding your domain knowledge.
# MAGIC 
# MAGIC **Flow of Modelling in prophet**:
# MAGIC - Installation of Prophet model.
# MAGIC - Splitting the data into training and testing sets.
# MAGIC - Renaming the date column to ds and target variable, pollution, to y
# MAGIC - Creating a multivariate prophet model using additional parameters(features)
# MAGIC - Fit your data and predict using the test data.
# MAGIC - Compare the predictions
# MAGIC 
# MAGIC **Reference Links**
# MAGIC - [**Facebook Prophet Introduction**](https://facebook.github.io/prophet/)
# MAGIC - [**FB Prophet Time Series**](https://www.kaggle.com/code/bagavathypriya/multivariate-time-series-using-fb-prophet)

# COMMAND ----------

!pip install jupyter_black

# COMMAND ----------

!pip install prophet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Installing the required libraries

# COMMAND ----------

# Importing the required libraries
# Libraries for reading the data and preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from prophet import Prophet

import warnings
warnings.filterwarnings("ignore")
import calendar
# pep8 formatting
import jupyter_black
jupyter_black.load()

# COMMAND ----------

# File location and type
try:
    file_location = "abfss://ml-suite@fiepmqasa.dfs.core.windows.net/HCE-Dublin/WTF/warehouse_hourly_data.csv"
    file_type = "csv"

    # CSV options
    infer_schema = "true"
    first_row_is_header = "true"
    delimiter = ","

    # The applied options are for CSV files. For other file types, these will be ignored.
    df_hourly = spark.read.format(file_type) \
      .option("inferSchema", infer_schema) \
      .option("header", first_row_is_header) \
      .option("sep", delimiter) \
      .load(file_location)
except Exception as e:
    raise FileNotFoundError("No such file found")

# COMMAND ----------

#Converting the Spark Dataframe into pandas dataframe
df_hourly = df_hourly.toPandas()
#Displaying the basic statistics
df_hourly.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation/ Insights:**
# MAGIC - The dataset contains hourly data for two months.
# MAGIC - It has 1441 rows and 33 features including the temporal KPIs.
# MAGIC - The dataset holds no missing values.
# MAGIC - Description for all the features of the dataset are given here:
# MAGIC [(**Hourly Data KPIs**)](https://confluence.honeywell.com/display/HCEHAL/%5BWIP%5D+Feature+Description+Hourly+Level+Dataset)

# COMMAND ----------

#Displaying a correlation matrix 
df_new = df_hourly.select_dtypes(include='number')
df_new = df_new.drop(columns=['Day','Month','Year'], axis =1)
corr = df_new.corr()
mask = np.zeros_like(corr, dtype=bool)
cmap = sns.diverging_palette(100, 7, s=75, l=40, n=5, center="light", as_cmap=True)

plt.figure(figsize=(20, 20))
sns.heatmap(corr,center=0, annot=True, fmt=".2f", square=True, cmap=cmap)
plt.show()

# COMMAND ----------

df_hourly.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Cycling encoding for Time related features.

# COMMAND ----------

def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
month1 = encode(df_hourly, 'Month', 12)
day1 = encode(df_hourly, 'Day', 365)
df_hourly['Weekday'] = df_hourly['Weekday'].astype(int)
df_hourly['Day of the week'] = df_hourly['Day of the week'].astype('category')
df_hourly['Day of the week'] = df_hourly['Day of the week'].cat.codes
day_of_the_week1 = encode(df_hourly, 'Day of the week', 7) 
df_hourly = df_hourly.drop(columns = ['Day', 'Month', 'Year','Day of the week'], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC **Lag Feature:**
# MAGIC Here, a new feature called 'Lag1_Picking_Efficiency' is introducted so that the model would forecast the Picking Efficinecy(Throughput) for the next hour(Lag1_Picking_Efficiency) as per the definition of ML-XO5. 

# COMMAND ----------

#Building a new target variable(Lag1_Picking_Efficiency)
df_hourly['Lag1_Picking_Efficiency'] = df_hourly['Picking_Efficiency'].shift(-1)
df_unseen = df_hourly[-1:]
df_hourly.drop(df_hourly.tail(1).index,inplace=True)
df_unseen.drop('Lag1_Picking_Efficiency', axis=1, inplace = True)
df_unseen

# COMMAND ----------

#PACF graph to determine the lag for the model.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rc("figure", figsize=(5, 3))
plot_pacf(df_hourly["Picking_Efficiency"], lags=30, method="ols")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Inference:**
# MAGIC - We will be forecasting picking efficiency for the next 1 hour, as per ML XO5 definition. Hence, we need to have the target variable for Picking Efficiency of the Next Hour. It is labelled as 'Lag1_Picking_Efficiency'.
# MAGIC - We can infer from the above graph that the Picking efficiency for the current hour is highly correlated to the Picking efficiency of the previous hour.
# MAGIC - Therefore, for our model we are considering the Lag=1.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3. Dividing the data into training and Testing sets.

# COMMAND ----------

df_hourly["Date"] = pd.to_datetime(
    df_hourly.Date, infer_datetime_format="True"
)
split_date = pd.datetime(2020, 11, 29)
train = df_hourly.loc[df_hourly.Date < split_date]
valid = df_hourly.loc[df_hourly.Date >= split_date]
valid.shape

# COMMAND ----------

# MAGIC %md
# MAGIC **Note on Training and validation dataset:**
# MAGIC - The training data consists of all the records from 01-10-2020 00:00:00 to 29-11-2020 00:00:00 with a total of 1417 records.
# MAGIC - The testing data consists of all the records from 29-11-2020 01:00:00 to 29-11-2020 23:00:00 with a total of 23 records.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Renaming the Date column into ds and Target variable to y.

# COMMAND ----------

train_sub = train
valid_sub = valid
valid2 = valid_sub.copy()
#We need to rename the date and target column as per the requirement for Prophet model.
train_sub.rename(columns={"Date": "ds", "Lag1_Picking_Efficiency": "y"}, inplace=True)
valid_sub.rename(columns={"Date": "ds", "Lag1_Picking_Efficiency": "y"}, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Building a Multivariate Prophet Model.

# COMMAND ----------

# MAGIC %md 
# MAGIC **Prophet Model:**
# MAGIC - By default, the prophet model has parameters for time series forecasting. But, for our use case, we have made yearly_seasonality = False as the data is only for two months.
# MAGIC - Also, the weekly_seasonality = True as there are more than 7 weeks of data.
# MAGIC - The parameter daily_seasonality = True as the granularity of the data is hourly level which means that every date has records for 24 hours.

# COMMAND ----------

model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,daily_seasonality = True,
    changepoint_prior_scale=0.001,
)
model.add_regressor('Day_sin')
model.add_regressor('Day_cos')
model.add_regressor('Month_sin')
model.add_regressor('Month_cos')
model.add_regressor('Day of the week_sin')
model.add_regressor('Day of the week_cos')
model.add_regressor('Weekday')
model.add_regressor("Total orders processed")
model.add_regressor("Number of units processed")
model.add_regressor("Units_Processed_Zone_1")
model.add_regressor("Units_Processed_Zone_2")
model.add_regressor("Old_Orders")
model.add_regressor("New_Orders")
model.add_regressor("Fulfilled_orders")
model.add_regressor("unfulfilled_orders")
model.add_regressor("partially_fulfilled_orders")
model.add_regressor("Domestic orders")
model.add_regressor("International orders")
model.add_regressor("Retail orders")
model.add_regressor("EachPicks")
model.add_regressor("CasePicks")
model.add_regressor("Perfect_Order_Rate")
model.add_regressor("Average_Grab_Time_in_minutes")
model.add_regressor("Average_Put_Time_in_minutes")
model.add_regressor("Total_chucks_used")
model.add_regressor("Total number of chuck trips done")
model.add_regressor("Labour_Headcount")
model.add_regressor("Labour_Headcount_Zone1")
model.add_regressor("Labour_Headcount_Zone2")
model.add_regressor("Labour_Time_in_minutes")
model.add_regressor("Labour_Time_Zone_1_in_minutes")
model.add_regressor("Labour_Time_Zone_2_in_minutes")
model.add_regressor("Avg_workload_per_worker_in_minutes")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Fitting the data and predict using the test data.

# COMMAND ----------

model.fit(train_sub)
forecast_multi_sub = model.predict(valid_sub.drop(columns="y"))
valid_sub = valid_sub.set_index("ds")
forecast_multi_sub = forecast_multi_sub.set_index("ds")

# COMMAND ----------

print('Min and Max Values of Lag picking efficiency in the whole data are:',round(df_hourly['Lag1_Picking_Efficiency'].min(),3),round(df_hourly['Lag1_Picking_Efficiency'].max(),3))
print('Min and Max Values of Lag picking efficiency in test data are: ',round(valid2['Lag1_Picking_Efficiency'].min(),3),round(valid2['Lag1_Picking_Efficiency'].max(),3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Compare the predictions.

# COMMAND ----------

valid_sub1 = valid_sub.merge(forecast_multi_sub.yhat, left_index=True, right_index=True)
valid_sub1.rename(columns = {'y':'Actual_Picking_Efficiency_for_next_hour','yhat':'Predicted_Picking_Efficiency_for_next_hour'},inplace = True)

# COMMAND ----------

import plotly.express as px
valid_sub1['Date'] = pd.date_range(start='2022-11-29', periods=24, freq = 'H')
fig = px.line(valid_sub1, x='Date', y=["Actual_Picking_Efficiency_for_next_hour", "Predicted_Picking_Efficiency_for_next_hour"], template = 'plotly_dark',
              labels={'ds':'Hour','value':'Picking Efficiency'} 
)
fig.update(layout_yaxis_range = [1.68 , 6.198])
fig.show()

# COMMAND ----------

valid_sub = valid_sub1
valid_sub.rename(columns = {'Actual_Picking_Efficiency_for_next_hour':'y','Predicted_Picking_Efficiency_for_next_hour':'yhat'},inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretations from the graph:**
# MAGIC - We can infer that model is showing good results for Predicted Picking efficiency and the actual picking efficiency.
# MAGIC - The min and max has been set on the y limit for better visualization and comparison of the Picking efficiency of the model.

# COMMAND ----------

def diagnostics(pred, valid):
    mse = np.mean(np.square(pred["yhat"] - valid["y"]))
    rmse = np.sqrt(mse)
    print("The RMSE is: ", rmse)
    mae = np.mean(np.abs(pred["yhat"] - valid["y"]))
    print("The MAE is: ", mae)

# COMMAND ----------

diagnostics(pred=forecast_multi_sub, valid=valid_sub)

# COMMAND ----------

def create_comparison_df(valid, forecast):
    picking_eff = valid.y.values
    predicted_picking_efficiency = forecast.yhat.values
    zipped = list(zip(picking_eff, predicted_picking_efficiency))
    columns = ["Lag1_Picking_Efficiency", "Predicted_Picking_Efficiency"]
    df = pd.DataFrame(zipped, columns=columns)
    return df

# COMMAND ----------

df_model = create_comparison_df(valid=valid_sub, forecast=forecast_multi_sub)
df_model.head(5)

# COMMAND ----------

#Displaying the mean of both actual and predicted picking efficiency. 
df_model.Lag1_Picking_Efficiency.mean(), df_model.Predicted_Picking_Efficiency.mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8. Saving the Final Model

# COMMAND ----------

from prophet.serialize import model_to_json, model_from_json
#Remove the existing model and replace it with a new trained model.
dbutils.fs.rm('abfss://ml-suite@fiepmqasa.dfs.core.windows.net/HCE-Dublin/WTF/serialised_prophet_model')
dbutils.fs.put('abfss://ml-suite@fiepmqasa.dfs.core.windows.net/HCE-Dublin/WTF/serialised_prophet_model',model_to_json(model)) 

# COMMAND ----------

dbutils.fs.ls('abfss://ml-suite@fiepmqasa.dfs.core.windows.net/HCE-Dublin/WTF/')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9. Prediction of the unseen data.

# COMMAND ----------

# MAGIC %md
# MAGIC **Explaination:**
# MAGIC - As seen in the previous steps, our validation dataset was considering all the records to forecast the first hour of 30-11-2020.
# MAGIC - In the below code, we have found a way to forecast the Picking efficiency for the next hour thereby handling the data leakage issue.
# MAGIC - The below code considers the Picking efficiency of the previous hour to forecast the next hour.

# COMMAND ----------

df_unseen.rename(columns={"Date": "ds"}, inplace=True)
df_unseen

# COMMAND ----------

#Predicting the thoughput for 30-11-2022 01:00:00 using the model created.
forecast_multi_sub1 = model.predict(df_unseen)

# COMMAND ----------

#Forecasted value for the next hour (30-11-2022 01:00:00)
forecast_multi_sub1.yhat

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10. Final Observation/Conclusion:
# MAGIC - The Prophet model performs very well for Multivariate Time series forecasting.
# MAGIC - The forecasted Picking efficiency for the next hour is almost closer to the previous hour.
# MAGIC - The overall mean of the forecast and the actual are closer.
