# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Technical notebook to generate visuals for Phase I
# MAGIC
# MAGIC I am re-using Erik's code to look at the 3 month subset

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import sum,avg,max,count

# COMMAND ----------

# See the names of the dataset provided.
mids261_mount_path      = "/mnt/mids-w261"
display(dbutils.fs.ls(f"{mids261_mount_path}/datasets_final_project_2022"))

# COMMAND ----------

# Read full dataset, get only daily data for NYC
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/")
df_NYC_weather = df_weather.filter(df_weather['NAME'] == 'JFK INTERNATIONAL AIRPORT, NY US')
df_to_plot = df_NYC_weather[['DATE','DailyAverageWindSpeed']].dropna()


# COMMAND ----------

# Convert weather to pandas, fix datatypes
df_to_plot = df_to_plot.toPandas()
df_to_plot['DATE'] = pd.to_datetime(df_to_plot['DATE'])
df_to_plot['DailyAverageWindSpeed'] = pd.to_numeric(df_to_plot['DailyAverageWindSpeed'], errors =  'coerce')
df_to_plot.dropna(inplace = True)
df_to_plot.columns = ['Date', 'Wind']
df_to_plot.head()

# COMMAND ----------

# Read full dataset, get only daily data for NYC
df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/")
JFK_codes = ['1247801', '1247802','1247803','1247804','1247805']
df_JFK_flights = df_flights.filter(df_flights['ORIGIN_AIRPORT_SEQ_ID'].isin(JFK_codes) )

# COMMAND ----------

delays_to_plot = df_JFK_flights[['FL_DATE','DEP_DELAY']].dropna()
delays_to_plot.head()

# COMMAND ----------

JFK_delays_daily = delays_to_plot.groupBy('FL_DATE') \
                    .agg(avg('DEP_DELAY').alias("avg_delay"), \
                        max('DEP_DELAY').alias("max_delay") ) \
                    .toPandas()
JFK_delays_daily.columns = ['Date', 'Mean_delay', 'Max_delay']

JFK_delays_daily.head()

# COMMAND ----------

JFK_delays_daily['Date'] = pd.to_datetime(JFK_delays_daily['Date'])

# COMMAND ----------

# Instantiate figure and axis
num_rows = 2
num_columns = 1
fig, axes = plt.subplots(num_rows, num_columns)
#Adjust space between plots in the figure
plt.subplots_adjust(hspace = 0.5)

#Fill the axis with data
axes[0].scatter(df_to_plot.Date, df_to_plot.Wind)  
axes[1].scatter(JFK_delays_daily.Date, JFK_delays_daily.Max_delay) 

#Set title and axis legend, only set axis legend on the sides
axes[0].set_title("Wind at JFK")
axes[1].set_title("Max delay")

# Remove the bounding box to make the graphs look less cluttered
#axes[0].spines['right'].set_visible(False)
#axes[0].spines['top'].set_visible(False)
#Format ticks
#for tick in axes[0].get_xticklabels():
#    tick.set_rotation(45)
plt.show()

# COMMAND ----------

# Reading the raw weather data to retrieve the column names. 
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")
weather_cols = df_weather.columns
print(f"There are {len(weather_cols)} weather columns in the combined dataset.")

# Reading in the combined dataset and filtering to just weather columns
df_otpw = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
df_otpw = df_otpw[weather_cols]
print(f"The 3 month subsample has {df_otpw.count()} total records.")


# COMMAND ----------



# COMMAND ----------

# sampling 10,000 records and converting to pandas df to quickly explore feature values
otpw = df_otpw.sample(fraction=10000/1401363).toPandas()

# COMMAND ----------

# creating table of feature describe()
feature_sample = otpw.describe().T

# quantifying a count threshold at 25% total
### since we are referencing a very small sample of data from a specific time period
### I do not what to put too much weight on how many nulls are present in this sample
### that is why I have the threshold at 25%. 
cnt_threshold = max(feature_sample['count'])*.25
print(f"Count threshold is {cnt_threshold}.")

# displaying records that are being removed
print("Below are the columns with less than 25% records present:")
potential_drops = feature_sample['count'][feature_sample['count'] < cnt_threshold].index.to_list()
print(potential_drops)


# COMMAND ----------

# based on a review of the features we are looking to omit due to limited records, 
# there are a series of 'Daily' metrics that could still be relevant and may not make the threshold 
# because they are only captured daily (1/24 < 25%).  Planning to re-include those in the list. 

# quantifying a list of the 'Daily' features
daily_cols = []
for column in potential_drops:
    if "Daily" in column:
        daily_cols.append(column)

# reviewing remaining feature list, sorted by count
updated_features = feature_sample[(feature_sample['count'] >= cnt_threshold) | (feature_sample.index.isin(daily_cols))]
updated_features = updated_features.sort_values(by='count', ascending=False)
print(f"There are {updated_features.shape[0]} features.")
updated_features

# COMMAND ----------

# creating a list of the columns
updated_feature_list = updated_features.index.to_list()

# filtering the original pandas 10k sample by these columns
otpw = otpw[updated_feature_list]

# function for trying to cast datatypes
def trycast(df):
    trycast_df = df.copy()
    for column in trycast_df.columns:
        try: # try to cast to int
            trycast_df[column] = trycast_df[column].astype(int)
        except (ValueError, TypeError):
            try: # try to cast to float
                trycast_df[column] = pd.to_numeric(trycast_df[column])
            except (ValueError, TypeError):
                pass # leave as string
    return trycast_df

# casting features to relevant datatypes
otpw = trycast(otpw)
# manually casting datetime types
otpw['DATE'] = pd.to_datetime(otpw['DATE'])
otpw['WindEquipmentChangeDate'] = pd.to_datetime(otpw['WindEquipmentChangeDate'])
data_types = pd.DataFrame(otpw.dtypes)
data_types

def map_dtypes(df):
    type_map = {'int64': 'integer', 'object': 'string', 'datetime64[ns]': 'datetime', 'float64': 'float'}
    return row['0']

# COMMAND ----------

display(otpw)

# COMMAND ----------


