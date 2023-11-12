# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Scratchbook for Erik's Contributions
# MAGIC I am using this file as the scratchbook for my contributions to the project. 
# MAGIC
# MAGIC ## 1) Initial Review of Weather Dataset
# MAGIC I am looking at the 3 month subset of the combined dataset and exploring the weather features present.

# COMMAND ----------

# importing modules for smaller summary datasets
import pandas as pd
import numpy as np

# COMMAND ----------

# Reading the raw weather data to retrieve the column names. 
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")
weather_cols = df_weather.columns
print(f"There are {len(weather_cols)} weather columns in the combined dataset.")

# Reading in the combined dataset and filtering to just weather columns
otpw = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
df_otpw = otpw[weather_cols]
print(f"The 3 month subsample has {df_otpw.count()} total records.")

# COMMAND ----------

# creating table of feature describe()
feature_sample = df_otpw.describe()

# converting feature sample to pandas df and transposing
feature_sample = feature_sample.toPandas().T

#promoting first row to headers, and dropping row
feature_sample.columns = feature_sample.iloc[0]
feature_sample = feature_sample.drop(feature_sample.index[0])
feature_sample['count'] = pd.to_numeric(feature_sample['count'])

# quantifying a count threshold at 25% total
threshold = 25.0
## since we are referencing a very small sample of data from a specific time period
## I do not what to put too much weight on how many nulls are present in this sample
## that is why I have the threshold at 25%. 
cnt_threshold = max(feature_sample['count'])*(threshold/100.0)
print(f"Count threshold is {cnt_threshold}.")

# displaying records that are being removed
print(f"Below are the columns with less than {threshold}% records present:")
potential_drops = feature_sample['count'][feature_sample['count'] < cnt_threshold]
potential_drops = pd.DataFrame(potential_drops).reset_index()
display(potential_drops)

# COMMAND ----------

# based on a review of the features we are looking to omit due to limited records, 
# there are a series of 'Daily' metrics that could still be relevant and may not make the threshold 
# because they are only captured daily (1/24 < 25%).  Planning to re-include those in the list. 

# quantifying a list of the 'Daily' features
daily_cols = []
for column in potential_drops['index']:
    if "Daily" in column:
        daily_cols.append(column)

# reviewing remaining feature list, sorted by count
updated_features = feature_sample[(feature_sample['count'] >= cnt_threshold) | (feature_sample.index.isin(daily_cols))]
updated_features = updated_features.sort_values(by='count', ascending=False)
print(f"There are {updated_features.shape[0]} features.")
updated_features

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Specific Exploration of Raw Weather Data

# COMMAND ----------

# selecting a specific day and location to see all observations recorded for it
example_data = df_weather[(df_weather['STATION'] =='72295023174') & (df_weather['DATE'].contains('2015-01-10'))]
display(example_data.sort('REPORT_TYPE'))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3) Checking if Weather Data is Always Departing Location

# COMMAND ----------

weather_loc_check = otpw[otpw['STATION']!=otpw['origin_station_id']]
display(weather_loc_check)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4) Reviewing Time-Based Components of Joined Dataset

# COMMAND ----------

lax_join_review = otpw['sched_depart_date_time', 'sched_depart_date_time_UTC', 'four_hours_prior_depart_UTC', 'two_hours_prior_depart_UTC', 'DATE', 'Report_Type'][otpw['STATION']=='72295023174']
display(lax_join_review)

# COMMAND ----------

###################### THIS SECTION IS BROKEN: SETUP FOR PANDAS BUT NEEDS TO BE FOR SPARK ##################################################################################

# creating a list of the columns
updated_feature_list = updated_features.index.to_list()

# filtering the original 3 month dataset by these columns
df_chosen_weather = df_otpw[updated_feature_list]

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
df_chosen_weather = trycast(df_chosen_weather)
# manually casting datetime types
df_chosen_weather['DATE'] = pd.to_datetime(df_chosen_weather['DATE'])
df_chosen_weather['WindEquipmentChangeDate'] = pd.to_datetime(df_chosen_weather['WindEquipmentChangeDate'])
data_types = pd.DataFrame(df_chosen_weather.dtypes)
data_types

# def map_dtypes(df):
#     type_map = {'int64': 'integer', 'object': 'string', 'datetime64[ns]': 'datetime', 'float64': 'float'}
#     return row['0']
