# Databricks notebook source
# MAGIC %md
# MAGIC #Phase 3: Notebook 2 - Imputation
# MAGIC The purpose of this notebook is as follows:

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import modules, data

# COMMAND ----------

# data processing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# importing visualization modules
import seaborn as sns
import matplotlib.pyplot as plt

# importing custom functions
from Code.funcs import blob_connect, get_df_dimensions, write_pd_df_to_storage
import funcs

# import pyspark modules
from pyspark.sql.functions import udf, isnan, isnull, when, count, col, regexp_extract, when, year, to_timestamp, countDistinct, collect_set, array, lit, array_except, collect_set, explode,to_date

from pyspark.sql import functions as F
from pyspark.ml.stat import Correlation
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

# import regex for parsing text columns
import re
import warnings

# filter append warning
warnings.filterwarnings('ignore', message='The frame.append method is deprecated and will be removed from pandas in a future version.')

# COMMAND ----------

# read in data from parquet
team_blob_url = funcs.blob_connect()
data_df = spark.read.parquet(f"{team_blob_url}/BK/clean_5yr_WITHOUT_2019").cache()

# COMMAND ----------

data_df.display()
data_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine Remaining Nulls to see what we can impute

# COMMAND ----------

# Function to get % null for each feature
def get_nulls(df):
    data_size = df.count()
    null_percents = df.select([((count(when(isnull(c), c)) / data_size) * 100).alias(c) for c in df.columns])
    null_percents_pd = null_percents.toPandas()
    
    return null_percents_pd

# get null % for a single feature
def get_null_percent_for_column(df, column_name):
    data_size = df.count()
    null_count = df.select(count(when(col(column_name).isNull(), column_name))).collect()[0][0]
    null_percent = (null_count / data_size) * 100
    
    return null_percent

# COMMAND ----------

# Create a dataframe of nulls
null_data = get_nulls(data_df).transpose().reset_index()
null_data.columns = ['Feature', '% Null']
null_data.sort_values(by="% Null")

# COMMAND ----------

# create a dataframe to track changes to nulls
null_impute_tracking = pd.DataFrame(columns=['Imputation','Null % Before', 'Null % After'])

# COMMAND ----------

# Create function to automatically update the dimension dataframe
def update_null_count_tracker(df, null_tracker_df, col_string, change_string):
    prev_null_count = null_data.loc[null_data['Feature'] == col_string, '% Null'].values[0]
    curr_null_count = get_null_percent_for_column(df, col_string)

    new_row = {"Imputation": change_string, "Null % Before": prev_null_count, "Null % After": curr_null_count}
    null_tracker_df = null_tracker_df.append(new_row, ignore_index=True)
    
    return null_tracker_df

# COMMAND ----------

# Look at nulls
review_nulls = null_data.sort_values(by="% Null", ascending = False)
review_nulls = review_nulls[(review_nulls['% Null'] < 80) & (review_nulls['% Null'] > 0)] #already filtered out + non-zero nulls
review_nulls.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Impute Data Based on Learnings

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impute `Snowfall` nulls as 0's
# MAGIC `dest_DailySnowfall` & `origin_DailySnowfall` are both highly relevant to flight delays, and based on the weather data documentation, we can make an argument for imputing these nulls as 0, since missing values are likely to indicate no snowfall.

# COMMAND ----------

snow_cols = ['origin_DailySnowfall', 'dest_DailySnowfall']

# impute snow columns as 0's
for snow_col in snow_cols:
    data_df = data_df.fillna(0, subset=snow_col)
    null_impute_tracking = update_null_count_tracker(data_df, null_impute_tracking, f'{snow_col}', f'{snow_col} as "0"')

null_impute_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impute `Precipitation` feature nulls as 0 
# MAGIC `origin_HourlyPrecipitation`,`dest_HourlyPrecipitation`,`origin_3Hr_Precipitation`,`dest_3Hr_Precipitation`,`origin_6Hr_Precipitation`,`dest_6Hr_Precipitation`,
# MAGIC `origin_12Hr_Precipitation`,`dest_12Hr_Precipitation`,`origin_DailyPrecipitation`,`dest_DailyPrecipitation` can be imputed as 0's. There is presumed to be no precipitation during that time period.

# COMMAND ----------

### this one takes a while to run

precip_impute_cols = ['origin_HourlyPrecipitation','dest_HourlyPrecipitation','origin_3Hr_Precipitation','dest_3Hr_Precipitation','origin_6Hr_Precipitation','dest_6Hr_Precipitation','origin_12Hr_Precipitation','dest_12Hr_Precipitation','origin_DailyPrecipitation','dest_DailyPrecipitation']

# impute precip columns as 0's
for precip_col in precip_impute_cols:
    data_df = data_df.fillna(0, subset=precip_col)
    null_impute_tracking = update_null_count_tracker(data_df, null_impute_tracking, f'{precip_col}', f'{precip_col} as "0"')

null_impute_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impute `DailyDepartureFromNormalAverageTemperature` feature nulls as 0
# MAGIC `origin_DailyDepartureFromNormalAverageTemperature` & `dest_DailyDepartureFromNormalAverageTemperature` can be imputed as 0's. There is presumed to be no daily departure from average during that time period.

# COMMAND ----------

dailydeptemp_impute_cols = ['origin_DailyDepartureFromNormalAverageTemperature','dest_DailyDepartureFromNormalAverageTemperature']

# impute depart from avg temp columns as 0's
for dep_avg in dailydeptemp_impute_cols:
    data_df = data_df.fillna(0, subset=dep_avg)
    null_impute_tracking = update_null_count_tracker(data_df, null_impute_tracking, f'{dep_avg}', f'{dep_avg} as "0"')

null_impute_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impute `PressureChange` feature nulls as 0 (no PressureChange in that time period)
# MAGIC pressureChange_impute_cols = `origin_3Hr_PressureChange`,`dest_3Hr_PressureChange`,`origin_6Hr_PressureChange`,`dest_6Hr_PressureChange`,`origin_12Hr_PressureChange`, and
# MAGIC `dest_12Hr_PressureChange` can be imputed as 0's. There is presumed to be no pressure change during that time period.

# COMMAND ----------

pressureChange_impute_cols = ['origin_3Hr_PressureChange','dest_3Hr_PressureChange','origin_6Hr_PressureChange','dest_6Hr_PressureChange','origin_12Hr_PressureChange','dest_12Hr_PressureChange']

# impute pressure as 0's
for press_col in dailydeptemp_impute_cols:
    data_df = data_df.fillna(0, subset=press_col)
    null_impute_tracking = update_null_count_tracker(data_df, null_impute_tracking, f'{press_col}', f'{press_col} as "0"')

null_impute_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pause and Review: Examine remaining nulls at this point

# COMMAND ----------

# columns imputed up to this point
imputed_cols = snow_cols + precip_impute_cols + dailydeptemp_impute_cols + pressureChange_impute_cols

# filter those out to get the current lis of nulls
updated_nulls = review_nulls[~review_nulls['Feature'].isin(imputed_cols)]
updated_nulls.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop na's for features with really low % nulls

# COMMAND ----------

#### this takes 20 minutes to execute: run with caution
low_null_perct = updated_nulls[updated_nulls['% Null'] < 1]['Feature'].tolist()

# impute pressure as 0's
for low_null_cols in low_null_perct:
    data_df = data_df.fillna(0, subset=low_null_cols)
    null_impute_tracking = update_null_count_tracker(data_df, null_impute_tracking, f'{low_null_cols}', f'{low_null_cols} nulls dropped')

null_impute_tracking

# COMMAND ----------

# review nulls again
review_nulls = review_nulls[(review_nulls['% Null'] < 80) & (review_nulls['% Null'] > 0)] #already filtered out + non-zero nulls
updated_nulls.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impute features as nearest days

# COMMAND ----------

# add column called calendar_date
# Add leading zeros to 'DAY_OF_MONTH' and 'DAY_OF_WEEK' columns if they are single digits
data_df = data_df.withColumn('DAY_OF_MONTH', F.lpad(col('DAY_OF_MONTH'), 2, '0'))
data_df = data_df.withColumn('DAY_OF_WEEK', F.lpad(col('DAY_OF_WEEK'), 2, '0'))

# Concatenate 'YEAR', 'DAY_OF_MONTH', and 'DAY_OF_WEEK' columns to form 'calendar_date' as string
data_df = data_df.withColumn('calendar_date',
    F.concat(
        col('YEAR'), lit('-'),
        col('DAY_OF_MONTH'), lit('-'),
        col('DAY_OF_WEEK')
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Test process on `Daily Avg Releative Humidity`

# COMMAND ----------

windowSpec = Window.orderBy(F.col("calendar_date").asc_nulls_last()) 

# fill nulls in the dest_DailyAverageRelativeHumidity column with the nearest non-null value
filled_data_df = data_df.withColumn(
    "filled_relative_humidity",
    F.when(
        F.col("dest_DailyAverageRelativeHumidity").isNull(),
        F.first(
            F.col("dest_DailyAverageRelativeHumidity"),
            ignorenulls=True
        ).over(windowSpec)
    ).otherwise(F.col("dest_DailyAverageRelativeHumidity"))
)

# COMMAND ----------

filled_data_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Repeat with Remaining Columns

# COMMAND ----------

columns_to_impute = [
    "dest_DailyAverageRelativeHumidity",
    "origin_DailyAverageStationPressure",
    "dest_DailyAverageStationPressure",
    "dest_DailySustainedWindSpeed",
    "origin_DailySustainedWindSpeed",
    "origin_DailySustainedWindDirection",
    "dest_DailySustainedWindDirection",
    "origin_DailyAverageRelativeHumidity",
    "dest_DailyAverageDryBulbTemperature",
    "origin_DailyAverageDryBulbTemperature"
]

# Function to fill with nearest non-null day value
def fill_nulls_with_nearest_value(df, column_name):
    windowSpec = Window.orderBy(F.col("calendar_date").asc_nulls_last()) 
    
    filled_column = (
        F.when(
            F.col(column_name).isNull(),
            F.first(
                F.col(column_name),
                ignorenulls=True
            ).over(windowSpec)
        ).otherwise(F.col(column_name)).alias(column_name)
    )
    
    return df.withColumn(column_name, filled_column)

# COMMAND ----------

# Apply the imputation function to each column
for column in columns_to_impute:
    data_df = fill_nulls_with_nearest_value(data_df, column)

# COMMAND ----------

data_df.display()

# COMMAND ----------

# update imputation tracker
for remain_col in columns_to_impute:
    null_impute_tracking = update_null_count_tracker(data_df, null_impute_tracking, f'{remain_col}', f'{remain_col} imputed as nearest non-null day value')

null_impute_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC #### Last Pass: Ensure no nulls are left

# COMMAND ----------

# Drop any features that should have been dropped
last_removes = [
    'dest_airport_name',
    'origin_airport_name',
    'DEST_AIRPORT_ID',
    'dest_icao',
    'origin_icao',
    'DEST_CITY_MARKET_ID',
    'OP_CARRIER_AIRLINE_ID',
    'OP_CARRIER',
    'ORIGIN_CITY_MARKET_ID',
    'OP_CARRIER_AIRLINE_ID',
    'OP_CARRIER',
    'origin_station_dis',
    'dest_station_dis',
    'DISTANCE_GROUP',
    'CRS_ARR_TIME',
    'ARR_TIME',
    'CRS_ELAPSED_TIME',
    'ACTUAL_ELAPSED_TIME',
    'AIR_TIME',
    'FL_DATE',
    'ORIGIN_AIRPORT_SEQ_ID',
    'ORIGIN_CITY_NAME',
    'ORIGIN_STATE_FIPS',
    'ORIGIN_STATE_NM',
    'ORIGIN_WAC',
    'DEST_AIRPORT_SEQ_ID',
    'DEST_CITY_NAME',
    'DEST_STATE_FIPS',
    'DEST_STATE_NM',
    'DEST_WAC',
    'WHEELS_OFF',
    'WHEELS_ON',
    'FLIGHTS',
    'FIRST_DEP_TIME',
    'TOTAL_ADD_GTIME',
    'LONGEST_ADD_GTIME',
    'origin_station_name',
    'origin_iata_code',
    'origin_region',
    'origin_station_lat',
    'origin_station_lon',
    'dest_station_name',
    'dest_iata_code',
    'dest_region',
    'dest_station_lat',
    'dest_station_lon',
    'four_hours_prior_depart_UTC'
]

# Filter the columns in last_removes to only include existing columns
need_to_drop = [col for col in last_removes if col in data_df.columns]
data_df = data_df.drop(*need_to_drop)

# COMMAND ----------

# Final pass to drop any remaining nulls
data_df = data_df.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Output: Imputed Features

# COMMAND ----------

#uncomment out when ready
data_df.write.parquet(f"{team_blob_url}/BK/clean_5yr_WITHOUT_2019_imputed")

# COMMAND ----------

# write imputation tracking dataframe to file
spark_df = spark.createDataFrame(null_impute_tracking)
spark_df.write.parquet(f"{team_blob_url}/BK/imputation_tracker")
