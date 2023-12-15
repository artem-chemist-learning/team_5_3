# Databricks notebook source
# MAGIC %md
# MAGIC #Phase 3: Notebook 1 - Data Integrity and Nulls
# MAGIC The purpose of this notebook is as follows:
# MAGIC   1. Review the 5-year dataset in entirety utilizing knowledge from EDA on 1-year dataset.
# MAGIC   2. Check the integrity of this data, e.g.
# MAGIC       - Are there variables with a disproportionate number of nulls in this dataset compared to the previous?
# MAGIC       - Do variable values change over time (e.g. certain airports disappear in later years)
# MAGIC   3. Re-evalate the data for variables to drop (nulls, variables lacking integrity)
# MAGIC   4. Prepare the dataset for the next stage in the pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import modules, data

# COMMAND ----------

# data processing
import pandas as pd
import numpy as np

# importing visualization modules
import seaborn as sns
import matplotlib.pyplot as plt

# importing custom functions
from Code.funcs import blob_connect, get_df_dimensions, write_pd_df_to_storage
import funcs

# import pyspark modules
from pyspark.sql.functions import udf, isnan, isnull, when, count, col, regexp_extract, when, year, to_timestamp, countDistinct, collect_set, array, lit, array_except, collect_set, explode
from pyspark.sql import functions as F
from pyspark.ml.stat import Correlation
from pyspark.sql.types import *
from pyspark.sql.window import Window

# import regex for parsing text columns
import re

# COMMAND ----------

# read in data from parquet
team_blob_url = funcs.blob_connect()
data_df = spark.read.parquet(f"{team_blob_url}/ES/new_joins/5Y").cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Automatically Track Changes to Data Dimensionality

# COMMAND ----------

# Create a dataframe to track data dimension changes as we go
data_dim_tracking = pd.DataFrame(columns=['Data Dimension Change','Cols Remaining', 'Rows Remaining'])

# COMMAND ----------

# Create function to automatically update the dimension dataframe
def update_dimension_tracker(df, dimension_df, change_string):
    cols, rows = get_df_dimensions(df)
    new_row = {"Data Dimension Change": change_string, "Cols Remaining": cols, "Rows Remaining": rows}
    dimension_df = dimension_df.append(new_row, ignore_index=True)
    return dimension_df

# COMMAND ----------

# Update tracking dataframe with initial data
data_dim_tracking = update_dimension_tracker(data_df, data_dim_tracking, "Initial Data Size")
data_dim_tracking

# COMMAND ----------

data_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filtering from Phase 1-2 Learnings
# MAGIC Before doing a deeper dive, we will filter out as many unusable or redundant columns as possible. This will allow us to narrow down our dataset before performing long data operations.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop Columns
# MAGIC These are columns we decided to drop during Phase 2 of this project. Brief reasoning is also provided.
# MAGIC
# MAGIC **NOTE:** These columns are provided as lists so that if we decide to remove others or add others back, we will simply be able to add or comment them out in this list. The data dictionary below is all-inclusive, and simply drops columns in this list.

# COMMAND ----------

# columns to be used for reference. keep for now, just calling them out here
ref_cols = [
    'ARR_DELAY',
    'ARR_DEL15',
    'DEP_TIME',
    'CRS_DEP_TIME',
    'TAXI_OUT',
    'DEP_DELAY',
    'TAXI_IN'
]

# columns that interfere with predictor or are imprecise
conflict_drop_cols = [
    'CARRIER_DELAY',                # conflicts with target
    'WEATHER_DELAY',                # conflicts with target
    'NAS_DELAY',                    # conflicts with target
    'SECURITY_DELAY',               # conflicts with target
    'LATE_AIRCRAFT_DELAY',          # conflicts with target
    'ARR_DELAY_NEW',                # less precise than ARR_DELAY
    'ARR_DELAY_GROUP',              # less granular than ARR_DELAY
    'ARR_TIME_BLK',                 # less granular than ARR_DELAY
    'DEP_DELAY_NEW',                # less precise than DEP_DELAY
    'DEP_DELAY_GROUP',              # less granular than DEP_DELAY
    'DEP_TIME_BLK',                 # less granular than DEP_DELAY
    'CANCELLATION_CODE']

# columns that were determined to have correlation with better-defined features
redundant_drop_cols = [
    'ORIGIN_CITY_MARKET_ID',
    'DEST_CITY_MARKET_ID',
    'DEST_AIRPORT_SEQ_ID',
    'ORIGIN_CITY_NAME',
    'DEST_CITY_NAME',
    'ORIGIN_STATE_FIPS',
    'ORIGIN_STATE_NM',              # full state name redunant to state abbrev. 
    'ORIGIN_WAC',                   # less granular than other loc. features 
    'DEST_STATE_NM',                # full state name redunant to state abbrev. 
    'DEST_STATE_FIPS',
    'DEST_WAC',                     # less granular than other loc. features 
    'dest_region',                  # one-to-one relationship with other location features
    'dest_station_lat',             # one-to-one relationship b/t ICAO & airport 
    'dest_station_lon',             # one-to-one relationship b/t ICAO & airport
    'dest_airport_lat',             # one-to-one relationship b/t ICAO & airport
    'dest_airport_lon',             # one-to-one relationship b/t ICAO & airport
    'dest_station_dis',             # one-to-one relationship b/t ICAO & airport
    'dest_UTC',                     # redundant after new join 
    'DISTANCE_GROUP',               # less granular than distance
    'WHEELS_OFF',                   # equivalent to DEP_TIME + TAXI_OUT
    'WHEELS_ON',                    # equivalent to ARR_TIME - TAXI_IN
    'origin_UTC',                   # redundant after new join 
    'dest_DATE',                    # redundant after new join 
    'FLIGHTS',                      # no additinal info provided from other cols
    'origin_airport_name',          # doesn't provide much info after reviewing International  vs. non-Int'l
    'origin_station_name',          # one-to-one relationship b/t station & airport
    'origin_station_id',            # one-to-one relationship b/t station & airport
    'origin_icao',                  # doesn't provide much info after reviewing International  vs. non-Int'l
    'origin_region',                # redundant to other loc. features 
    'origin_region',                # redundant to other loc. features 
    'origin_station_dis',           # one-to-one relationship b/t station & airport
    'dest_airport_name',            # doesn't provide much info after reviewing International  vs. non-Int'l
    'dest_station_name',            # one-to-one relationship b/t station & airport
    'dest_station_id',              # one-to-one relationship b/t ICAO & airport 
    'dest_icao',                    # doesn't provide much info after reviewing International  vs. non-Int'l
    'four_hours_prior_depart_UTC',  # redundant after new join
    'three_hours_prior_depart_UTC', # redundant after new join 
    'origin_DATE',                  # redundant after new join 
    'dest_UTC',                     # redundant after new join
    'FL_DATE',                      # redundant to other time columns
    'OP_CARRIER_AIRLINE_ID',
    'OP_CARRIER',
    'AIR_TIME',
    'ACTUAL_ELAPSED_TIME',
    'ARR_TIME'
]

# Drop columns from list
data_df = data_df.drop(*conflict_drop_cols)
data_df = data_df.drop(*redundant_drop_cols)

# COMMAND ----------

# Review current shape after drop
data_dim_tracking = update_dimension_tracker(data_df, data_dim_tracking, "Drop redundant / imprecise columns")
data_dim_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filter By Canceled
# MAGIC Conceptually, canceled flights are not what we are trying to measure, as cancellations are different than delays. For this reason, we have decided to remove them from our data.

# COMMAND ----------

data_df = data_df[data_df['CANCELLED'] < 1.0]

# COMMAND ----------

# Review current shape after drop
data_dim_tracking = update_dimension_tracker(data_df, data_dim_tracking, "Drop cancelled flights")
data_dim_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop data missing target variable
# MAGIC If we do not have the target variable, we will not be able to use it for prediction.

# COMMAND ----------

data_df = data_df.na.drop(subset=['DEP_DEL15'])

# COMMAND ----------

# Review current shape after drop
data_dim_tracking = update_dimension_tracker(data_df, data_dim_tracking, "Drop flights without target variable")
data_dim_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ## Re-Evaluate Nulls: Drop > 80% Null
# MAGIC Though we already have a list of features that had too many nulls for usage in the first 2 phases, we re-perform this analysis to check for more features that must be dropped, or see if any features could be usable in the larger set.

# COMMAND ----------

# Function to get % null for each feature
def get_nulls(df):
    data_size = df.count()
    #null_percents = df.select([(100.0 * count(when(col(c).isNull(), c))/data_size).alias(c) for c in df.columns])
    null_percents = df.select([((count(when(isnull(c), c)) / data_size) * 100).alias(c) for c in df.columns])
    null_percents_pd = null_percents.toPandas()
    
    return null_percents_pd

# COMMAND ----------

# Function to get the columns with > 80% Nulls
null_data = get_nulls(data_df).transpose().reset_index()
null_data.columns = ['Feature', '% Null']
null_data.sort_values(by="% Null")

# COMMAND ----------

# Drop Columns with > 80% nulls
nulls_to_drop = null_data.loc[null_data['% Null'] > 80, 'Feature'].tolist()
data_df = data_df.drop(*nulls_to_drop)

# Display col names
# Note: We got the same columns to drop as we did before, in addition to Wind Gust data 
print(nulls_to_drop)

# COMMAND ----------

# Review current shape after drop
data_dim_tracking = update_dimension_tracker(data_df, data_dim_tracking, "Drop columns with > 80% nulls")
data_dim_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema: Cast Datatypes

# COMMAND ----------

# define function to cast spark df columsn to new type(s) and apply to 3-month dataset
def cast_cols_to_new_type(spark_df, columns_to_cast):
    # define the expressions for each column
    cast_exprs = [f"cast({col} as {new_type}) as {col}" for col, new_type in columns_to_cast.items()]
    
    # apply the cast expressions
    spark_df = spark_df.selectExpr(*cast_exprs)

    return spark_df


#### ALL possible columns are in this dictionary. I then systematically remove columns from out DATAFRAME that we don't want.
#### This way, we will easily be able to add columns back or remove columns by modifying the list and preserving the dictionary
    # create dictionary of ALL column types (note: casting only selected cols to change will omit them from resulting df)
new_coltypes_dict = {'QUARTER': 'integer', #can drop this unless we want to somehow use it for partitioning
                     'DAY_OF_MONTH': 'integer', 
                     'DAY_OF_WEEK': 'integer',
                     'FL_DATE': 'date', 
                     'OP_UNIQUE_CARRIER': 'string', # if not consistent over time (b/c some airlines fold or merge), check against 2 features below
                     'OP_CARRIER_AIRLINE_ID': 'integer', # should be redundant to other 2 carrier ID features - check before dropping
                     'OP_CARRIER': 'string', # should be redundant to other 2 carrier ID features - check before dropping
                     'TAIL_NUM': 'string', 
                     'OP_CARRIER_FL_NUM': 'string',
                     'ORIGIN_AIRPORT_ID': 'string',
                     'ORIGIN_AIRPORT_SEQ_ID': 'string', # should probably use these to check consistency of 'DEST_AIRPORT_ID" over time 
                     'ORIGIN_CITY_MARKET_ID': 'string',
                     'ORIGIN': 'string',
                     'ORIGIN_CITY_NAME': 'string', #not using this unless we want to group by city 
                     'ORIGIN_STATE_ABR': 'string', # not sure yet if really useful - check in EDA/DT
                     'ORIGIN_STATE_FIPS': 'integer',
                     'ORIGIN_STATE_NM': 'string', # excluded - full state name redunant to state abbrev. 
                     'ORIGIN_WAC': 'integer',  # excluded - less granular than other loc. features (basically a continent code)
                     'DEST_AIRPORT_ID': 'string',
                     'DEST_AIRPORT_SEQ_ID': 'integer', # should probably use these to check consistency of 'DEST_AIRPORT_ID" over time 
                     'DEST_CITY_MARKET_ID': 'integer', # excluded b/c no info gain (only a few city markets have multiple airports)
                     'DEST': 'string',
                     'DEST_CITY_NAME': 'string',  #not using this unless we want to group by city 
                     'DEST_STATE_ABR': 'string', # not sure yet if really useful - check in EDA/DT
                     'DEST_STATE_FIPS': 'integer', # excluded - redundant to STATE
                     'DEST_STATE_NM': 'string',# excluded - full state name redunant to state abbrev. 
                     'DEST_WAC': 'integer', # excluded - less granular than other loc. features (basically a continent code)
                     'CRS_DEP_TIME': 'float', # scheduled departure time (local time)
                     'DEP_TIME': 'float',  # actual departure time (local time)
                     'DEP_DELAY': 'double',
                     'DEP_DELAY_NEW': 'double',  # excluded b/c less precise than DEP_DELAY (whihc includes neg. vals for early flights)
                     'DEP_DEL15': 'double', 
                     'DEP_DELAY_GROUP': 'integer', #dep delay  interval, dropped b/c less granular 
                     'DEP_TIME_BLK': 'string', #dep delay  interval, dropped b/c less granular 
                     'TAXI_OUT': 'double',
                     'WHEELS_OFF': 'float', # excluded in 1yr - eqiivalent to DEP_TIME + TAXI_OUT
                     'WHEELS_ON': 'float', # excluded in 1yr - eqiivalent to ARR_TIME - TAXI_IN
                     'TAXI_IN': 'double',
                     'CRS_ARR_TIME': 'integer',
                     'ARR_TIME': 'integer',
                     'ARR_DELAY': 'double',
                     'ARR_DELAY_NEW': 'double',  # excluded b/c less precise than ARR_DELAY (whihc includes neg. vals for early flights)
                     'ARR_DEL15': 'double', # not sure if we're using for FE
                     'ARR_DELAY_GROUP': 'integer', #arr delay  interval, dropped b/c less granular 
                     'ARR_TIME_BLK': 'string', #arr delay  interval, dropped b/c less granular 
                     'CANCELLED': 'double', # --> drop AFTER removing null obs 
                     'CANCELLATION_CODE': 'string', # excluded 
                     'DIVERTED': 'double', 
                     'CRS_ELAPSED_TIME': 'double', 
                     'ACTUAL_ELAPSED_TIME': 'double', 
                     'AIR_TIME': 'double', 
                     'FLIGHTS': 'float', # excluded b/c no info gain (each row = 1)
                     'DISTANCE': 'double',
                     'DISTANCE_GROUP': 'integer', # ecluded b/c less granular than distance
                     'CARRIER_DELAY': 'double', #excluded due to 80+% nulls in 1-year
                     'WEATHER_DELAY': 'double', #excluded due to 80+% nulls in 1-year
                     'NAS_DELAY': 'double', #excluded due to 80+% nulls in 1-year
                     'SECURITY_DELAY': 'double', #excluded due to 80+% nulls in 1-year
                     'LATE_AIRCRAFT_DELAY': 'double', #excluded due to 80+% nulls in 1-year
                     'FIRST_DEP_TIME': 'integer', #excluded due to 90+% nulls in 1-year
                     'TOTAL_ADD_GTIME': 'double', #excluded due to 90+% nulls in 1-year
                     'LONGEST_ADD_GTIME': 'double', #excluded due to 90+% nulls in 1-year
                     'YEAR': 'integer', 
                     'MONTH': 'integer', # redundant - this info contained in scheduled departure UTC
                     'origin_airport_name': 'string', #  #excluded after determining International  vs. non-Int'l. doesn't provide much info
                     'origin_station_name': 'string', #excluded after determining one-to-one relationship b/t station & airport
                     'origin_station_id': 'bigint', #excluded after determining one-to-one relationship b/t station & airport
                     'origin_iata_code': 'string',
                     'origin_icao': 'string',
                     'origin_type': 'string',
                     'origin_region': 'string', #excluded after determining this was redundant to other loc. features 
                     'origin_station_lat': 'double', 
                     'origin_station_lon': 'double', 
                     'origin_airport_lat': 'double',
                     'origin_airport_lon': 'double',
                     'origin_station_dis': 'double', #excluded after determining one-to-one relationship b/t station & airport
                     'dest_airport_name': 'string', #excluded after determining International  vs. non-Int'l. doesn't provide much info
                     'dest_station_name': 'string', #excluded after determining one-to-one relationship b/t station & airport
                     'dest_station_id': 'bigint', #excluded after determining one-to-one relationship b/t station & airport
                     'dest_iata_code': 'string',
                     'dest_icao': 'string', #excluded after determining one-to-one relationship b/t ICAO & airport 
                     'dest_type': 'string',
                     'dest_region': 'string', #excluded after determining this was redundant to other loc. features 
                     'dest_station_lat': 'double', #excluded after determining one-to-one relationship b/t ICAO & airport 
                     'dest_station_lon': 'double', #excluded after determining one-to-one relationship b/t ICAO & airport 
                     'dest_airport_lat': 'double', #excluded after determining one-to-one relationship b/t ICAO & airport 
                     'dest_airport_lon': 'double', #excluded after determining one-to-one relationship b/t ICAO & airport 
                     'dest_station_dis': 'double', #excluded after determining one-to-one relationship b/t station & airport
                     'sched_depart_date_time_UTC': 'timestamp', # --> use this to engineer 'day_of_year' ()
                     'four_hours_prior_depart_UTC': 'timestamp', # don't need - redundant after new join
                     'two_hours_prior_depart_UTC': 'timestamp',
                     'origin_DailySnowfall': 'float',
                     'origin_DailyPrecipitation': 'float',
                     'origin_DailyDepartureFromNormalAverageTemperature': 'float',
                     'origin_DailyAverageDryBulbTemperature': 'float',
                     'origin_DailyAverageRelativeHumidity': 'float',
                     'origin_DailyAverageStationPressure': 'float',
                     'origin_DailySustainedWindDirection': 'float',
                     'origin_DailySustainedWindSpeed': 'float',
                     'dest_DailySnowfall': 'float',
                     'dest_DailyPrecipitation': 'float',
                     'dest_DailyDepartureFromNormalAverageTemperature': 'float',
                     'dest_DailyAverageDryBulbTemperature': 'float',
                     'dest_DailyAverageRelativeHumidity': 'float',
                     'dest_DailyAverageStationPressure': 'float',
                     'dest_DailySustainedWindDirection': 'float',
                     'dest_DailySustainedWindSpeed': 'float',
                     'three_hours_prior_depart_UTC': 'timestamp', # don't need - redundant after new join 
                     'origin_DATE': 'date',# don't need - redundant after new join 
                     'origin_HourlyDryBulbTemperature': 'float',
                     'origin_HourlyStationPressure': 'float',
                     'origin_HourlyPressureChange': 'float',
                     'origin_HourlyWindGustSpeed': 'float',
                     'origin_HourlyWindDirection': 'float',
                     'origin_HourlyPrecipitation': 'float',
                     'origin_HourlyVisibility': 'float',
                     'origin_3Hr_DryBulbTemperature': 'double',
                     'origin_6Hr_DryBulbTemperature': 'double',
                     'origin_12Hr_DryBulbTemperature': 'double',
                     'origin_3Hr_PressureChange': 'double',
                     'origin_6Hr_PressureChange': 'double',
                     'origin_12Hr_PressureChange': 'double',
                     'origin_3Hr_StationPressure': 'double',
                     'origin_6Hr_StationPressure': 'double',
                     'origin_12Hr_StationPressure': 'double',
                     'origin_3Hr_WindGustSpeed': 'double',
                     'origin_6Hr_WindGustSpeed': 'double',
                     'origin_12Hr_WindGustSpeed': 'double',
                     'origin_3Hr_Precipitation': 'double',
                     'origin_6Hr_Precipitation': 'double',
                     'origin_12Hr_Precipitation': 'double',
                     'origin_3Hr_Visibility': 'double',
                     'origin_6Hr_Visibility': 'double',
                     'origin_12Hr_Visibility': 'double',
                     'origin_UTC': 'timestamp', # don't need - redundant after new join 
                     'dest_DATE': 'date', # don't need - redundant after new join 
                     'dest_HourlyDryBulbTemperature': 'float',
                     'dest_HourlyStationPressure': 'float',
                     'dest_HourlyPressureChange': 'float',
                     'dest_HourlyWindGustSpeed': 'float',
                     'dest_HourlyWindDirection': 'float',
                     'dest_HourlyPrecipitation': 'float',
                     'dest_HourlyVisibility': 'float',
                     'dest_3Hr_DryBulbTemperature': 'double',
                     'dest_6Hr_DryBulbTemperature': 'double',
                     'dest_12Hr_DryBulbTemperature': 'double',
                     'dest_3Hr_PressureChange': 'double',
                     'dest_6Hr_PressureChange': 'double',
                     'dest_12Hr_PressureChange': 'double',
                     'dest_3Hr_StationPressure': 'double',
                     'dest_6Hr_StationPressure': 'double',
                     'dest_12Hr_StationPressure': 'double',
                     'dest_3Hr_WindGustSpeed': 'double',
                     'dest_6Hr_WindGustSpeed': 'double',
                     'dest_12Hr_WindGustSpeed': 'double',
                     'dest_3Hr_Precipitation': 'double',
                     'dest_6Hr_Precipitation': 'double',
                     'dest_12Hr_Precipitation': 'double',
                     'dest_3Hr_Visibility': 'double',
                     'dest_6Hr_Visibility': 'double',
                     'dest_12Hr_Visibility': 'double',
                     'dest_UTC': 'timestamp' # don't need - redundant after new join 
                     }

# remove keys corresponding to dropped columns (known drops + null drops)
dropped_columns = conflict_drop_cols + redundant_drop_cols + nulls_to_drop
new_coltypes_dict = {col: col_type for col, col_type in new_coltypes_dict.items() if col not in dropped_columns}

# COMMAND ----------

#Cast and Review New Types
casted_data = cast_cols_to_new_type(data_df, new_coltypes_dict)
casted_data.display()
casted_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable Integrity over time
# MAGIC We want to check whether certain variables vary over time in their values. For example, if all of a sudden in 2017 and onward the airline carrier "Hawaiian Airlines" simply disappears from the dataset, that is something we would like to know.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Carrier Consistency 

# COMMAND ----------

# cast 'YEAR' to timestamp and extract the year directly: will use this for all data below
df = data_df.withColumn('YEAR', col('YEAR').cast('timestamp'))

# COMMAND ----------

# DBTITLE 1,Consistency of Carrier ID
# group by year and distinct carrier values for each year
# note: you can sort by year by clicking the arrow on the table
grouped_carrier = df.groupBy('YEAR').agg(collect_set('OP_UNIQUE_CARRIER').alias('unique_carriers'))

# look at 2015-2018
filtered_carriers = grouped_carrier.filter((year('YEAR') >= 2015) & (year('YEAR') <= 2018))
filtered_carriers.display(truncate=False)

# COMMAND ----------

# airline name reference table if it helps:
airline_names_df = spark.read.parquet(f"{team_blob_url}/airline_names_df")#.toPandas()
display(airline_names_df)

# COMMAND ----------

# DBTITLE 1,Carrier IDs that dropped off
# need to explode per row
exploded_unique_carriers = grouped_carrier.withColumn('exploded_carriers', explode('unique_carriers'))

# get missing carriers for each year by comparing with all possible carriers
missing_carriers_per_year = exploded_unique_carriers.join(
    airline_names_df,
    on=exploded_unique_carriers['exploded_carriers'] == airline_names_df['Code'],
    how='left').filter(col('Code').isNull()).groupBy('YEAR').agg(
    collect_set('exploded_carriers').alias('missing_carriers'))

# look at 2015-2018
filtered_missing_carriers = missing_carriers_per_year.filter((year('YEAR') >= 2015) & (year('YEAR') <= 2018))
filtered_missing_carriers.display()

# COMMAND ----------

# DBTITLE 1,Consistency of Airline ID
# per lucy: 
"""
'OP_UNIQUE_CARRIER': 'string', # if not consistent over time (b/c some airlines fold or merge), check against 2 features below
'OP_CARRIER_AIRLINE_ID': 'integer', # should be redundant to other 2 carrier ID features - check before dropping
'OP_CARRIER': 'string', # should be redundant to other 2 carrier ID features - check before dropping
"""

# group by year and distinct carrier values for each year
# note: you can sort by year by clicking the arrow on the table
grouped_airline_id = df.groupBy('YEAR').agg(collect_set('OP_CARRIER_AIRLINE_ID').alias('unique_airline_id'))
grouped_airline_id.display(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Flight Number and Airport Origin Consistency 

# COMMAND ----------

# group by year and distinct ORIGIN_AIRPORT_ID for each year
grouped_airport_id = df.groupBy('YEAR').agg(collect_set('ORIGIN_AIRPORT_ID').alias('unique_airport_id'))
grouped_airport_id.display(truncate=False)

# COMMAND ----------

# group by year and distinct OP_CARRIER_FL_NUM for each year
grouped_flight_num = df.groupBy('YEAR').agg(collect_set('OP_CARRIER_FL_NUM').alias('unique_flight_nums')) 
grouped_flight_num.display(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine Remaining Nulls to see what we can drop

# COMMAND ----------

few_nulls = null_data.sort_values(by="% Null", ascending = False)
few_nulls = few_nulls[(few_nulls['% Null'] < 80) & (few_nulls['% Null'] > 0)] #already filtered out + non-zero nulls
few_nulls.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop wind gust features: only have 6 / 12 hr values, still a lopt of nulls

# COMMAND ----------

wind_features = 'origin_6Hr_WindGustSpeed', 'dest_6Hr_WindGustSpeed', 'dest_12Hr_WindGustSpeed', 'origin_12Hr_WindGustSpeed'
casted_data = casted_data.drop(*wind_features)

# COMMAND ----------

# update data dimension tracker
data_dim_tracking = update_dimension_tracker(casted_data, data_dim_tracking, "Drop wind features")
data_dim_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC #### Can drop hourly pressure change only reported every 3 hours, so hourly feature is useless

# COMMAND ----------

pressure_features = 'origin_HourlyPressureChange', 'dest_HourlyPressureChange'
casted_data = casted_data.drop(*pressure_features)

# COMMAND ----------

# update data dimension tracker
data_dim_tracking = update_dimension_tracker(casted_data, data_dim_tracking, "Drop hourly pressure features")
data_dim_tracking

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check how many flights are affected by airline carriers that drop off

# COMMAND ----------

# filter to get rows with carrier id that drops off in 2018
carriers_to_check = ["YX", "YV", "OH", "G4", "9E"]
carrier_drop_off_flights = casted_data.filter(col('OP_UNIQUE_CARRIER').isin(carriers_to_check))
count_flights = carrier_drop_off_flights.count()
count_flights

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Storage

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split 2019 data (hold-out set out of this) so that it doesn't get touched

# COMMAND ----------

# filter data for years 2015 to 2018
dataset_2015_to_2018 = casted_data.filter((col("YEAR") >= 2015) & (col("YEAR") <= 2018))

# Filter data for the year 2019
dataset_2019 = casted_data.filter(col("YEAR") == 2019)

# COMMAND ----------

# switched to overwrite
dataset_2019.write.mode("overwrite").parquet(f"{team_blob_url}/BK/clean_2019_ONLY")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write rest of data (without 2019) to its own file

# COMMAND ----------

# switched to overwrite
dataset_2015_to_2018.write.mode("overwrite").parquet(f"{team_blob_url}/BK/clean_5yr_WITHOUT_2019")

# COMMAND ----------

# check that writing to storage works: 2015-2018
cleaned_5yr_df_without_2019 = spark.read.parquet(f"{team_blob_url}/BK/clean_5yr_WITHOUT_2019").cache()
cleaned_5yr_df_without_2019.display()
cleaned_5yr_df_without_2019.printSchema()

# COMMAND ----------

# check that writing to storage works: 2019
cleaned_5yr_2019 = spark.read.parquet(f"{team_blob_url}/BK/clean_2019_ONLY").cache()
cleaned_5yr_2019.display()
cleaned_5yr_2019.printSchema()

# COMMAND ----------

print("2015 - 2018 Size: ", cleaned_5yr_df_without_2019.count())
print("2019 Size: ", cleaned_5yr_2019.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Keep a version of the dataset that is not split (full 5 year)

# COMMAND ----------

casted_data.write.mode("overwrite").parquet(f"{team_blob_url}/BK/clean_5yr")

# COMMAND ----------

# write dimension tracking dataframe to file
spark_df = spark.createDataFrame(data_dim_tracking)
spark_df.write.parquet(f"{team_blob_url}/BK/dimension_tracker")
