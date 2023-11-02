# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Technical notebook to generate visuals for Phase I
# MAGIC
# MAGIC I am re-using Erik's code to look at the 3 month subset

# COMMAND ----------

import pandas as pd
import numpy as np

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

# Get a panda's dataframe with the daily windspeed
df_to_plot = df_weather[['DATE','DailyAverageWindSpeed']].toPandas()
df_to_plot['DATE'] = pd.to_datetime(df_to_plot['DATE'])
df_to_plot['DailyAverageWindSpeed'] = df_to_plot['DailyAverageWindSpeed'].astype('float')
df_to_plot.dropna(inplace = True)
df_to_plot.head()

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


