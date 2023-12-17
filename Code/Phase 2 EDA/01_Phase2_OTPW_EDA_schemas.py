# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 EDA: "New join" OTPW Data (3 month & 1 year)
# MAGIC
# MAGIC **Team 5-3: Bailey Kuehl, Lucy Herr, Artem Lebedev, Erik Sambraillo** 
# MAGIC <br>**261 Fall 2023**
# MAGIC <br>
# MAGIC <br>
# MAGIC This notebook applies the comprehensive schemas for weather and flights features after the datasets were re-joined for completeness.
# MAGIC

# COMMAND ----------

# importing modules for smaller summary datasets
import pandas as pd
import numpy as np

# importing visualization modules
import seaborn as sns
import matplotlib.pyplot as plt
#import folium 

# importing custom functions
import funcs

# import pyspark modules
from pyspark.sql import functions as F
from pyspark.ml.stat import Correlation
from pyspark.sql.types import *
from pyspark.sql.window import Window

# import regex for parsing text columns
import re

# COMMAND ----------

# connect to team blob
team_blob_url = funcs.blob_connect()
# view blob storage root folder 
display(dbutils.fs.ls(f"{team_blob_url}/"))

# COMMAND ----------

# reviewing the project blob storage
mids261_mount_path      = "/mnt/mids-w261"
display(dbutils.fs.ls(f"{mids261_mount_path}/datasets_final_project_2022/"))

# COMMAND ----------

otpw_3m_newjoin = spark.read.parquet(f"{team_blob_url}/ES/new_joins/3MO")

# COMMAND ----------

# MAGIC %md
# MAGIC The datatypes for the weather columns in the newly joined data appear correct, but all of the flights variables (through `two_hours_prior_depart_UTC`) are string type.

# COMMAND ----------

# check schema for 1-year new join data:
otpw_1yr_newjoin = spark.read.parquet(f"{team_blob_url}/ES/new_joins/1YR")
otpw_1yr_newjoin.printSchema()

# COMMAND ----------

# check original version (raw) of otwp 3-month data
otpw_3m_orig = spark.read.parquet(f"{team_blob_url}/OTPW_3M_raw/")

# COMMAND ----------

# check dimensions for each dataframe to evaluate whether we can merge them 
funcs.get_df_dimensions(otpw_3m_newjoin)
#funcs.get_df_dimensions(otpw_3m_orig)

# COMMAND ----------

# MAGIC %md
# MAGIC The number of rows (observations) don't align between the datasets, so we can't do a simple merge. 
# MAGIC Instead, we need to specify the schema for the new dataframe. 

# COMMAND ----------

otpw_3m_orig.printSchema

# COMMAND ----------

# combine otpw_3m_orig.schema[0:90] with otpw_newjoin_3m.schema[91:] (apologies for hardcoding due to the time crunch)
# from pyspark.sql.types import StructType,StructField,IntegerType,DateType,StringType, LongType, DoubleType, FloatType, TimestampType
# full_schema = StructType([
#     StructField('QUARTER', IntegerType(), True), 
#     StructField('DAY_OF_MONTH', IntegerType(), True), 
#     StructField('DAY_OF_WEEK', IntegerType(), True), 
#     StructField('FL_DATE', DateType(), True), 
#     StructField('OP_UNIQUE_CARRIER', StringType(), True), 
#     StructField('OP_CARRIER_AIRLINE_ID', IntegerType(), True), 
#     StructField('OP_CARRIER', StringType(), True), 
#     StructField('TAIL_NUM', StringType(), True), 
#     StructField('OP_CARRIER_FL_NUM', IntegerType(), True), 
#     StructField('ORIGIN_AIRPORT_ID', IntegerType(), True), 
#     StructField('ORIGIN_AIRPORT_SEQ_ID', IntegerType(), True), 
#     StructField('ORIGIN_CITY_MARKET_ID', IntegerType(), True), 
    # StructField('ORIGIN', StringType(), True), StructField('ORIGIN_CITY_NAME', StringType(), True), StructField('ORIGIN_STATE_ABR', StringType(), True), StructField('ORIGIN_STATE_FIPS', IntegerType(), True), StructField('ORIGIN_STATE_NM', StringType(), True), StructField('ORIGIN_WAC', IntegerType(), True), StructField('DEST_AIRPORT_ID', IntegerType(), True), StructField('DEST_AIRPORT_SEQ_ID', IntegerType(), True), StructField('DEST_CITY_MARKET_ID', IntegerType(), True), StructField('DEST', StringType(), True), StructField('DEST_CITY_NAME', StringType(), True), StructField('DEST_STATE_ABR', StringType(), True), StructField('DEST_STATE_FIPS', IntegerType(), True), StructField('DEST_STATE_NM', StringType(), True), StructField('DEST_WAC', IntegerType(), True), StructField('CRS_DEP_TIME', IntegerType(), True), StructField('DEP_TIME', IntegerType(), True), StructField('DEP_DELAY', DoubleType(), True), StructField('DEP_DELAY_NEW', DoubleType(), True), StructField('DEP_DEL15', DoubleType(), True), StructField('DEP_DELAY_GROUP', IntegerType(), True), StructField('DEP_TIME_BLK', StringType(), True), StructField('TAXI_OUT', DoubleType(), True), StructField('WHEELS_OFF', IntegerType(), True), StructField('WHEELS_ON', IntegerType(), True), StructField('TAXI_IN', DoubleType(), True), StructField('CRS_ARR_TIME', IntegerType(), True), StructField('ARR_TIME', IntegerType(), True), StructField('ARR_DELAY', DoubleType(), True), StructField('ARR_DELAY_NEW', DoubleType(), True), StructField('ARR_DEL15', DoubleType(), True), StructField('ARR_DELAY_GROUP', IntegerType(), True), StructField('ARR_TIME_BLK', StringType(), True), StructField('CANCELLED', DoubleType(), True), StructField('CANCELLATION_CODE', StringType(), True), StructField('DIVERTED', DoubleType(), True), StructField('CRS_ELAPSED_TIME', DoubleType(), True), StructField('ACTUAL_ELAPSED_TIME', DoubleType(), True), StructField('AIR_TIME', DoubleType(), True), StructField('FLIGHTS', DoubleType(), True), StructField('DISTANCE', DoubleType(), True), StructField('DISTANCE_GROUP', IntegerType(), True), StructField('CARRIER_DELAY', DoubleType(), True), StructField('WEATHER_DELAY', DoubleType(), True), StructField('NAS_DELAY', DoubleType(), True), StructField('SECURITY_DELAY', DoubleType(), True), StructField('LATE_AIRCRAFT_DELAY', DoubleType(), True), StructField('FIRST_DEP_TIME', IntegerType(), True), StructField('TOTAL_ADD_GTIME', DoubleType(), True), StructField('LONGEST_ADD_GTIME', DoubleType(), True), StructField('YEAR', IntegerType(), True), StructField('MONTH', IntegerType(), True), StructField('origin_airport_name', StringType(), True), StructField('origin_station_name', StringType(), True), StructField('origin_station_id', LongType(), True), StructField('origin_iata_code', StringType(), True), StructField('origin_icao', StringType(), True), StructField('origin_type', StringType(), True), StructField('origin_region', StringType(), True), StructField('origin_station_lat', DoubleType(), True), StructField('origin_station_lon', DoubleType(), True), StructField('origin_airport_lat', DoubleType(), True), StructField('origin_airport_lon', DoubleType(), True), StructField('origin_station_dis', DoubleType(), True), StructField('dest_airport_name', StringType(), True), StructField('dest_station_name', StringType(), True), StructField('dest_station_id', LongType(), True), StructField('dest_iata_code', StringType(), True), StructField('dest_icao', StringType(), True), StructField('dest_type', StringType(), True), StructField('dest_region', StringType(), True), StructField('dest_station_lat', DoubleType(), True), StructField('dest_station_lon', DoubleType(), True), StructField('dest_airport_lat', DoubleType(), True), StructField('dest_airport_lon', DoubleType(), True), StructField('dest_station_dis', DoubleType(), True), StructField('sched_depart_date_time', TimestampType(), True), StructField('sched_depart_date_time_UTC', TimestampType(), True), StructField('origin_DailySnowfall', FloatType(), True), StructField('origin_DailyPrecipitation', FloatType(), True), StructField('origin_DailyDepartureFromNormalAverageTemperature', FloatType(), True), StructField('origin_DailyAverageDryBulbTemperature', FloatType(), True), StructField('origin_DailyAverageRelativeHumidity', FloatType(), True), StructField('origin_DailyAverageStationPressure', FloatType(), True), StructField('origin_DailySustainedWindDirection', FloatType(), True), StructField('origin_DailySustainedWindSpeed', FloatType(), True), StructField('dest_DailySnowfall', FloatType(), True), StructField('dest_DailyPrecipitation', FloatType(), True), StructField('dest_DailyDepartureFromNormalAverageTemperature', FloatType(), True), StructField('dest_DailyAverageDryBulbTemperature', FloatType(), True), StructField('dest_DailyAverageRelativeHumidity', FloatType(), True), StructField('dest_DailyAverageStationPressure', FloatType(), True), StructField('dest_DailySustainedWindDirection', FloatType(), True), StructField('dest_DailySustainedWindSpeed', FloatType(), True), StructField('three_hours_prior_depart_UTC', TimestampType(), True), 
    # StructField('origin_DATE', StringType(), True), StructField('origin_HourlyDryBulbTemperature', FloatType(), True), StructField('origin_HourlyStationPressure', FloatType(), True), StructField('origin_HourlyPressureChange', FloatType(), True), StructField('origin_HourlyWindGustSpeed', FloatType(), True), StructField('origin_HourlyWindDirection', FloatType(), True), StructField('origin_HourlyPrecipitation', FloatType(), True), StructField('origin_HourlyVisibility', FloatType(), True), StructField('origin_3Hr_DryBulbTemperature', DoubleType(), True), StructField('origin_6Hr_DryBulbTemperature', DoubleType(), True), StructField('origin_12Hr_DryBulbTemperature', DoubleType(), True), StructField('origin_3Hr_PressureChange', DoubleType(), True), StructField('origin_6Hr_PressureChange', DoubleType(), True), StructField('origin_12Hr_PressureChange', DoubleType(), True), StructField('origin_3Hr_StationPressure', DoubleType(), True), StructField('origin_6Hr_StationPressure', DoubleType(), True), StructField('origin_12Hr_StationPressure', DoubleType(), True), StructField('origin_3Hr_WindGustSpeed', DoubleType(), True), StructField('origin_6Hr_WindGustSpeed', DoubleType(), True), StructField('origin_12Hr_WindGustSpeed', DoubleType(), True), StructField('origin_3Hr_Precipitation', DoubleType(), True), StructField('origin_6Hr_Precipitation', DoubleType(), True), StructField('origin_12Hr_Precipitation', DoubleType(), True), StructField('origin_3Hr_Visibility', DoubleType(), True), StructField('origin_6Hr_Visibility', DoubleType(), True), StructField('origin_12Hr_Visibility', DoubleType(), True), StructField('origin_UTC', TimestampType(), True), StructField('dest_DATE', StringType(), True), StructField('dest_HourlyDryBulbTemperature', FloatType(), True), StructField('dest_HourlyStationPressure', FloatType(), True), StructField('dest_HourlyPressureChange', FloatType(), True), StructField('dest_HourlyWindGustSpeed', FloatType(), True), StructField('dest_HourlyWindDirection', FloatType(), True), StructField('dest_HourlyPrecipitation', FloatType(), True), StructField('dest_HourlyVisibility', FloatType(), True), StructField('dest_3Hr_DryBulbTemperature', DoubleType(), True), StructField('dest_6Hr_DryBulbTemperature', DoubleType(), True), StructField('dest_12Hr_DryBulbTemperature', DoubleType(), True), StructField('dest_3Hr_PressureChange', DoubleType(), True), StructField('dest_6Hr_PressureChange', DoubleType(), True), StructField('dest_12Hr_PressureChange', DoubleType(), True), StructField('dest_3Hr_StationPressure', DoubleType(), True), StructField('dest_6Hr_StationPressure', DoubleType(), True), StructField('dest_12Hr_StationPressure', DoubleType(), True), StructField('dest_3Hr_WindGustSpeed', DoubleType(), True), StructField('dest_6Hr_WindGustSpeed', DoubleType(), True), StructField('dest_12Hr_WindGustSpeed', DoubleType(), True), StructField('dest_3Hr_Precipitation', DoubleType(), True), StructField('dest_6Hr_Precipitation', DoubleType(), True), StructField('dest_12Hr_Precipitation', DoubleType(), True), StructField('dest_3Hr_Visibility', DoubleType(), True), StructField('dest_6Hr_Visibility', DoubleType(), True), StructField('dest_12Hr_Visibility', DoubleType(), True), StructField('dest_UTC', TimestampType(), True)
    ])

# COMMAND ----------

# orig_cols = set(otpw_3m_orig.columns)
# newjoin_cols = set(otpw_newjoin_3m.columns)
# #print(f"New join columns not in original OTPW dataset: {newjoin_cols.difference(orig_cols)}")
# print(f"Original columns not in new join dataset: {orig_cols.difference(newjoin_cols)}")

# COMMAND ----------

# reload new join otpw 3month data with full (correct) schema - not wokring

otpw_3m_newjoin_sch = spark.read.parquet(f"{team_blob_url}/ES/new_joins/3MO", schema=full_schema, header=True)

# COMMAND ----------

# define function to cast spark df columsn to new type(s) and apply to 3-month dataset
def cast_cols_to_new_type(spark_df, columns_to_cast):
    # define the expressions for each column
    cast_exprs = [f"cast({col} as {new_type}) as {col}" for col, new_type in columns_to_cast.items()]
    # apply the cast expressions
    spark_df = spark_df.selectExpr(*cast_exprs)
    #spark_df.printSchema()
    return spark_df

# create dictionary of ALL column types 
# (note: casting only selected cols to change will omit them from resulting df)
new_coltypes_dict = {'QUARTER': 'integer', 
                     'DAY_OF_MONTH': 'integer', 
                     'DAY_OF_WEEK': 'integer',
                     'FL_DATE': 'date',
                     'OP_UNIQUE_CARRIER': 'string',
                     'OP_CARRIER_AIRLINE_ID': 'integer',
                     'OP_CARRIER': 'string', 
                     'TAIL_NUM': 'string', 
                     'OP_CARRIER_FL_NUM': 'string',
                     'ORIGIN_AIRPORT_ID': 'string',
                     'ORIGIN_AIRPORT_SEQ_ID': 'string',
                     'ORIGIN_CITY_MARKET_ID': 'string',
                     'ORIGIN': 'string',
                     'ORIGIN_CITY_NAME': 'string',
                     'ORIGIN_STATE_ABR': 'string',
                     'ORIGIN_STATE_FIPS': 'integer',
                     'ORIGIN_STATE_NM': 'string',
                     'ORIGIN_WAC': 'integer',
                     'DEST_AIRPORT_ID': 'string',
                     'DEST_AIRPORT_SEQ_ID': 'integer',
                     'DEST_CITY_MARKET_ID': 'integer',
                     'DEST': 'string',
                     'DEST_CITY_NAME': 'string',
                     'DEST_STATE_ABR': 'string', 
                     'DEST_STATE_FIPS': 'integer',
                     'DEST_STATE_NM': 'string',
                     'DEST_WAC': 'integer',
                     'CRS_DEP_TIME': 'float', 
                     'DEP_TIME': 'float',  
                     'DEP_DELAY': 'double',
                     'DEP_DELAY_NEW': 'double', 
                     'DEP_DEL15': 'double', 
                     'DEP_DELAY_GROUP': 'integer',
                     'DEP_TIME_BLK': 'string', 
                     'TAXI_OUT': 'double',
                     'WHEELS_OFF': 'float',
                     'WHEELS_ON': 'float',
                     'TAXI_IN': 'double',
                     'CRS_ARR_TIME': 'integer',
                     'ARR_TIME': 'integer',
                     'ARR_DELAY': 'double',
                     'ARR_DELAY_NEW': 'double',
                     'ARR_DEL15': 'double',
                     'ARR_DELAY_GROUP': 'integer',
                     'ARR_TIME_BLK': 'string',
                     'CANCELLED': 'double',
                     'CANCELLATION_CODE': 'string',
                     'DIVERTED': 'double',
                     'CRS_ELAPSED_TIME': 'double',
                     'ACTUAL_ELAPSED_TIME': 'double',
                     'AIR_TIME': 'double',
                     'FLIGHTS': 'float', 
                     'DISTANCE': 'double',
                     'DISTANCE_GROUP': 'integer',
                     'CARRIER_DELAY': 'double',
                     'WEATHER_DELAY': 'double', 
                     'NAS_DELAY': 'double',
                     'SECURITY_DELAY': 'double',
                     'LATE_AIRCRAFT_DELAY': 'double',
                     'FIRST_DEP_TIME': 'integer',
                     'TOTAL_ADD_GTIME': 'double',
                     'LONGEST_ADD_GTIME': 'double',
                     'YEAR': 'integer',
                     'MONTH': 'integer',
                     'origin_airport_name': 'string',
                     'origin_station_name': 'string',
                     'origin_station_id': 'bigint',
                     'origin_iata_code': 'string',
                     'origin_icao': 'string',
                     'origin_type': 'string',
                     'origin_region': 'string',
                     'origin_station_lat': 'double', 
                     'origin_station_lon': 'double', 
                     'origin_airport_lat': 'double',
                     'origin_airport_lon': 'double',
                     'origin_station_dis': 'double',
                     'dest_airport_name': 'string',
                     'dest_station_name': 'string',
                     'dest_station_id': 'bigint',
                     'dest_iata_code': 'string',
                     'dest_icao': 'string',
                     'dest_type': 'string',
                     'dest_region': 'string',
                     'dest_station_lat': 'double', 
                     'dest_station_lon': 'double', 
                     'dest_airport_lat': 'double',
                     'dest_airport_lon': 'double',
                     'dest_station_dis': 'double',
                     'sched_depart_date_time_UTC': 'timestamp',
                     'four_hours_prior_depart_UTC': 'timestamp',
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
                     'three_hours_prior_depart_UTC': 'timestamp',
                     'origin_DATE': 'date',
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
                     'origin_UTC': 'timestamp',
                     'dest_DATE': 'date',
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
                     'dest_UTC': 'timestamp'}

otwp_1yr_newjoin_schema = cast_cols_to_new_type(otpw_1yr_newjoin, new_coltypes_dict)

# COMMAND ----------

funcs.get_df_dimensions(otwp_1yr_newjoin_schema)

# COMMAND ----------

otwp_1yr_newjoin_schema.printSchema()

# COMMAND ----------

display(otwp_1yr_newjoin_schema)

# COMMAND ----------

# write file with corrected schema as parquet file to team storage as "1YR_schema"
#otwp_1yr_newjoin_schema.write.parquet(f"{team_blob_url}/ES/new_joins/1YR_schema")
otwp_1yr_newjoin_schema.write.mode("overwrite").parquet(f"{team_blob_url}/ES/new_joins/1YR_schema")

# COMMAND ----------

# to correct previously overwriting original 3-month 'new join' data 
# w/ a df containing only the selected (type-corrected) columns, 
# select the 1st 3 months of 2015 data from the unmodified 1-year "new join" data
# & write to storage

filtered_df = otpw_1yr_newjoin.filter(
    (F.col("year") == 2015) & 
    (F.col("month").isin([1, 2, 3]))
)

funcs.get_df_dimensions(filtered_df)

# COMMAND ----------

# confirm the number of rows in the filtered df aligns with previous file 
funcs.get_df_dimensions(otpw_3m_newjoin)

# COMMAND ----------

# re-write unmodified 3-month "new join" parquet file to team storage
filtered_df.write.mode("overwrite").parquet(f"{team_blob_url}/ES/new_joins/3MO")

# COMMAND ----------

# now, write 3-month "new join" parquet file w/ modified schema to team storage
otpw_3m_newjoin_schema = cast_cols_to_new_type(filtered_df, new_coltypes_dict)
otpw_3m_newjoin_schema.printSchema


# COMMAND ----------

otpw_3m_newjoin_schema.write.parquet(f"{team_blob_url}/ES/new_joins/3MO_schema")

