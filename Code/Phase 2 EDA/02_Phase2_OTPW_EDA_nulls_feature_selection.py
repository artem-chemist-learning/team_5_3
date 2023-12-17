# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 EDA: Re-joined 1-year OTPW Dataset
# MAGIC
# MAGIC **Team 5-3: Bailey Kuehl, Lucy Herr, Artem Lebedev, Erik Sambraillo** 
# MAGIC <br>**261 Fall 2023**
# MAGIC <br>
# MAGIC <br>
# MAGIC This notebook (02) is the second step of the full EDA process for the 1-year OTPW dataset (2015). This version of the data has been re-joined from the raw flights and weather tables in order to retain a higher proportion of the original weather feature values.
# MAGIC
# MAGIC Overview of EDA & data cleaning steps performed in this notebook:
# MAGIC 1. Address missing data for all features and drop or impute features with high proportions of null values
# MAGIC 2. Perform initial feature selection: remove redundant or irrelevant features and extract non-modeling categorical and text features for airport and weather stations ID lookup tables 
# MAGIC 3. Evaluate and remove cancelled flight observations 
# MAGIC

# COMMAND ----------

# importing modules for smaller summary datasets
import pandas as pd
import numpy as np

# importing visualization modules
import seaborn as sns
import matplotlib.pyplot as plt

# importing custom functions
import funcs

# import pyspark modules
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.stat import Correlation
from pyspark.sql.types import *
from pyspark.sql.window import Window


# COMMAND ----------

# connect to team blob
team_blob_url = funcs.blob_connect()
# view blob storage root folder 
display(dbutils.fs.ls(f"{team_blob_url}/LH"))

# COMMAND ----------

# load re-joined 1-year OTPW data with corrected schema 
otpw_1y = spark.read.parquet(f"{team_blob_url}/ES/new_joins/1YR_schema")
# view dimensions of 1-year joined data
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

display(otpw_1y)

# COMMAND ----------

# separate last quarter of 2015 data for hold-out set? (October-December 2015)
# otpw_Q4 = otpw_1y.filter(
#     (F.col("YEAR") == 2015) & 
#     (F.col("MONTH").isin([10, 11, 12]))
# )
# funcs.get_df_dimensions(otpw_Q4)

# otpw_Q123 = otpw_1y.filter(
#     (F.col("YEAR") == 2015) & 
#     ~(F.col("MONTH").isin([10, 11, 12]))
# )
# funcs.get_df_dimensions(otpw_Q123)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check missing data by feature 
# MAGIC - drop features with 90% null values or greater
# MAGIC - determine imputation approaches for remaining features 

# COMMAND ----------

# calculate missing values per column 
#missing_df = funcs.count_missing_values(otpw_1y)

# write missing df to blob (more efficient to load again than re-run)
# missing_df.reset_index(inplace=True)
# missing_df.rename(columns={'index': 'column_name'}, inplace=True)
# missing_sdf = spark.createDataFrame(missing_df)
# missing_sdf.write.mode('overwrite').parquet(f"{team_blob_url}/LH/p2_missing_sdf")

# load missing values df per column dataframe and convert to pandas
missing_sdf = spark.read.parquet(f"{team_blob_url}/LH/p2_missing_sdf")
missing_df = missing_sdf.toPandas()
missing_df.head(20)

# COMMAND ----------

# fix schema (? this was never an issue on previous rounds )
# define the schema
# missing_schema = StructType([
#     StructField("feature_name", StringType(), True),
#     StructField("null_percent", FloatType(), True),
# ])

# Read the Parquet file using the defined schema
#missing_sdf = spark.read.schema(missing_schema).parquet(f"{team_blob_url}/LH/p2_missing_sdf")

# missing_coltypes_dict = {'column_name': 'string', 'null_percent': 'float'}
# missing_sdf = funcs.cast_cols_to_new_type(missing_sdf, missing_coltypes_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop features with 90+% nulls (4)

# COMMAND ----------

# MAGIC %md
# MAGIC `FIRST_DEP_TIME`,`LONGEST_ADD_GTIME`,`TOTAL_ADD_GTIME`,`CANCELLATION_CODE`

# COMMAND ----------

# drop columns with 90%+ nulls from otpw 1 year 
columns_to_drop = ['LONGEST_ADD_GTIME','TOTAL_ADD_GTIME','FIRST_DEP_TIME','CANCELLATION_CODE']
funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*columns_to_drop)
funcs.get_df_dimensions(otpw_1y)


# COMMAND ----------

# inspect features w/ 80-90% nulls
funcs.filter_df_by_min_max(missing_df, 'null_percent', 80, 90)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Specific delay type features (5) 
# MAGIC `CARRIER_DELAY`,`LATE_AIRCRAFT_DELAY`,`SECURITY_DELAY`,`NAS_DELAY',& 'WEATHER_DELAY`

# COMMAND ----------

# inspect specific delay features in the 80-90% null range: 
# # 'CARRIER_DELAY','LATE_AIRCRAFT_DELAY','SECURITY_DELAY','NAS_DELAY',& 'WEATHER_DELAY' 

delay_cols = ['DEP_DELAY','DEP_DELAY_NEW','CARRIER_DELAY','LATE_AIRCRAFT_DELAY','SECURITY_DELAY','NAS_DELAY','WEATHER_DELAY']

funcs.plot_boxplots(otpw_1y, delay_cols, sample_fraction=0.1)


# COMMAND ----------

funcs.compare_specific_delay_values(otpw_1y,'CARRIER_DELAY')

# COMMAND ----------

#compare_specific_delay_values(otpw_1y,'WEATHER_DELAY')
funcs.compare_specific_delay_values(otpw_1y,'WEATHER_DELAY')

# COMMAND ----------

#compare_specific_delay_values(otpw_1y,'LATE_AIRCRAFT_DELAY')
funcs.compare_specific_delay_values(otpw_1y,'LATE_AIRCRAFT_DELAY')

# COMMAND ----------

#compare_specific_delay_values(otpw_1y,'WEATHER_DELAY')
funcs.compare_specific_delay_values(otpw_1y,'WEATHER_DELAY')

# COMMAND ----------

#compare_specific_delay_values(otpw_1y,'SECURITY_DELAY')
funcs.compare_specific_delay_values(otpw_1y,'SECURITY_DELAY')

# COMMAND ----------

#compare_specific_delay_values(otpw_1y,'NAS_DELAY')
funcs.compare_specific_delay_values(otpw_1y,'NAS_DELAY')

# COMMAND ----------

# MAGIC %md
# MAGIC **Decisions re: CARRIER_DELAY, LATE_AIRCRAFT_DELAY, SECURITY_DELAY, NAS_DELAY, & WEATHER_DELAY**:<br>
# MAGIC As the sparse specific delay columns contain the same information as the general "DEP_DELAY" feature, we can make an argument for discarding them at this stage. 

# COMMAND ----------

delay_drop_cols = ['CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'SECURITY_DELAY',
                   'NAS_DELAY','WEATHER_DELAY']
funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*delay_drop_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's continue examining the relatively sparse features that have slightly lower percentages of nulls: 

# COMMAND ----------

# features w/ 60-80% nulls
funcs.filter_df_by_min_max(missing_df, 'null_percent', 60,80)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Wind gust speed features (8): drop
# MAGIC `dest_HourlyWindGustSpeed`, `origin_HourlyWindGustSpeed`, `dest_3Hr_WindGustSpeed`, & `origin_3Hr_WindGustSpeed`

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In our phase 2 modeling, we can safely drop the WindGustSpeeds features given that these values have not yet been engineered to include information on wind direction. Our background research has indicated that commercial flight performance is relatively unaffected by airport winds (or may even benefit from it to some extent). The correct way to feature engineer these wind speed variables would be to create vectors using both wind and direction - we will revisit the possibility of this as time permits in Phase 3. 

# COMMAND ----------

missing_df[missing_df['column_name'].str.contains('WindGust')]

# COMMAND ----------

# wind gust speed features in the 60-90% null percent range : 
# these are less  likely to contribute useful information (as they are decoupled from direction). we can make an argument for dropping them at this stage and revisiting in later modeling. 
wind_gust_feats = missing_df[missing_df['column_name'].str.contains('WindGust')]['column_name']
wind_gust_feats = wind_gust_feats.tolist()

funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*wind_gust_feats)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Hourly Pressure Change features (2)
# MAGIC `dest_HourlyPressureChange` & `origin_HourlyPressureChange`

# COMMAND ----------

# MAGIC %md
# MAGIC  
# MAGIC These features were engineered by Erik through re-joining the weather and flights data. After reviewing the weather data documentation, he determined that pressure change is only recorded every three hours in the data, so the 3, 6, & 12 hour representations are more precise. The hourly features can be dropped without losing information (as opposed to being imputed). 
# MAGIC

# COMMAND ----------

missing_df[missing_df['column_name'].str.contains('HourlyPressureChange')]

# COMMAND ----------

# drop destination & origin hourly pressure change features 
hourly_pressure_feats = missing_df[missing_df['column_name'].str.contains('HourlyPressureChange')]['column_name'].tolist()

funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*hourly_pressure_feats )
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features with 10-60% null values 
# MAGIC Now let's look at the features with between 10-60% missing values: 

# COMMAND ----------

# features w/ 10-60% nulls
funcs.filter_df_by_min_max(missing_df, 'null_percent', 10,60)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Daily Snowfall features (2): impute 
# MAGIC
# MAGIC `dest_DailySnowfall` & `origin_DailySnowfall`

# COMMAND ----------

# MAGIC %md
# MAGIC These two snowfall features are both highly relevant to flight delays, and based on the weather data documentation, we can make an argument for imputing these nulls as 0, since missing values are likely to indicate no snowfall. 
# MAGIC

# COMMAND ----------

# impute null values in 'dest_DailySnowfall' & 'origin_DailySnowfall' with 0
print(f"Number of nulls in 'dest_DailySnowfall' before imputing: {otpw_1y.filter(F.col('dest_DailySnowfall').isNull()).count()}")
print(f"Number of nulls in 'origin_DailySnowfall' before imputing: {otpw_1y.filter(F.col('origin_DailySnowfall').isNull()).count()}")

otpw_1y = otpw_1y.withColumn('origin_DailySnowfall',\
    F.coalesce(F.col('origin_DailySnowfall'), F.lit(0)))\
        .withColumn('dest_DailySnowfall',\
            F.coalesce(F.col('dest_DailySnowfall'), F.lit(0)))
        
print(f"Number of nulls in 'dest_DailySnowfall' after imputing: {otpw_1y.filter(F.col('dest_DailySnowfall').isNull()).count()}")
print(f"Number of nulls in 'origin_DailySnowfall' after imputing: {otpw_1y.filter(F.col('origin_DailySnowfall').isNull()).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Remaining features with nulls
# MAGIC Finally, let's check the remaining features with any null values (between 1 and 20%)

# COMMAND ----------

# features w/ 1-20% nulls
funcs.filter_df_by_min_max(missing_df, 'null_percent', 1,20)
#  DELAY/ARR nulls need to be inspected and potentially imputed (or dropped) --> do in next phase of EDA while looking at specific flight/weather features

# COMMAND ----------

# features w/ 0-1% nulls
funcs.filter_df_by_min_max(missing_df, 'null_percent',0.0001,1)

# COMMAND ----------

# MAGIC %md
# MAGIC Other than `origin_DailyAverageRelativeHumidity` and `dest_DailyAverageRelativeHumidity`, which each have ~15% nulls, all remaining features have under 5%. We will hold off on determining how to address the weather features with nulls until we look at them more closely in the next step of this EDA. In general, the weather data is more sparse, even after the new join, so we would't expect it to be without some missing data. 
# MAGIC
# MAGIC However, it's more concerning that some subsets of the original flights features have the same (albeit small) proportions of missing values. Because the flights data is generally very complete (which makes sense givben the high degree of regulation and coordination over air travel data), we do need into these subsets of observations because these patterns are likely meaningful.  

# COMMAND ----------

#1.81% nulls: ['ARR_DELAY','ARR_DELAY_GROUP','ACTUAL_ELAPSED_TIME','ARR_DEL15','ARR_DELAY_NEW','AIR_TIME'] 	 

null_df = otpw_1y.select(['OP_UNIQUE_CARRIER','ORIGIN_AIRPORT_ID','TAIL_NUM','FL_DATE','CANCELLED','DIVERTED','ARR_DELAY','ARR_DELAY_GROUP','ACTUAL_ELAPSED_TIME','ARR_DEL15','ARR_DELAY_NEW','AIR_TIME']).filter(F.col('ARR_DELAY').isNull()).describe()
null_df.toPandas()

# COMMAND ----------

# total rows in subset 
n_null_arrdelay = otpw_1y.select('ARR_DELAY').filter(F.col('ARR_DELAY').isNull()).count()

# count cancelled flights in this subset
null_arrdelay_cxl = ((F.col('ARR_DELAY').isNull()) & (F.col('CANCELLED')==1))
n_cxl = otpw_1y.select('ARR_DELAY').filter(null_arrdelay_cxl).count()

# count diverted flights in this subset
null_arrdelay_div = ((F.col('ARR_DELAY').isNull()) & (F.col('DIVERTED')==1))
n_div = otpw_1y.select('ARR_DELAY').filter(null_arrdelay_div).count()

print(f"N cancelled flights {n_cxl} + N diverted flights  {n_div} = {n_cxl + n_div}")
n_cxl + n_div == n_null_arrdelay

# COMMAND ----------

# MAGIC %md
# MAGIC Given that this subset contains only cancelled and diverted flights, we can certainly make the case for dropping at least the cancelled flights since won't have associated delay values and therefore can't contribute predictive value to our models. First, let's check whether this subset aligns with the other missing flights values.
# MAGIC - 1.59% nulls: `WHEELS_ON`,`TAXI_IN`, `ARR_TIME`
# MAGIC - 1.48% nulls: `DEP_DELAY`,`DEP_DELAY_GROUP`,`DEP_DEL15`,`DEP_DELAY_NEW`,`DEP_TIME`
# MAGIC - 0.25% nulls: `TAIL_NUM`
# MAGIC - 0.0001% nulls:`CRS_ELAPSED_TIME`
# MAGIC

# COMMAND ----------

# total cancelled flights for reference 
total_cxl = otpw_1y.select('CANCELLED').filter(F.col('CANCELLED')==1).count()
print(f"total cancelled flights:  {total_cxl}")
# total diverted flights for reference 
total_div = otpw_1y.select('DIVERTED').filter(F.col('DIVERTED')==1).count()
print(f"total diverted flights:  {total_div}")

# COMMAND ----------

null_df = otpw_1y.select(['OP_UNIQUE_CARRIER','ORIGIN_AIRPORT_ID','TAIL_NUM',
                          'FL_DATE','CANCELLED','DIVERTED','ARR_DELAY',
                          'ARR_DELAY_GROUP','ACTUAL_ELAPSED_TIME',
                          'ARR_DEL15','ARR_DELAY_NEW','AIR_TIME']).filter(F.col('ARR_DELAY').isNull()).describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remove cancelled flight observations

# COMMAND ----------

# check "cancelled" flight observations (null tail number & carrier info )
cancelled_df = otpw_1y.filter(otpw_1y['CANCELLED'] == 1)
cancelled_summary = cancelled_df.describe()
cancelled_summary.toPandas()

# COMMAND ----------

print(f"Percentage of cancelled flight observations: {100*cancelled_df.count()/otpw_1y.count()} ")

# COMMAND ----------

# drop these cancelled flight observations as they constitute only a small proportion of the overall data (as well not being associated with values of the predicted delay outcome)
funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.filter(
    ~(F.col("CANCELLED") == 1))
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

otpw_1y_select_cols_df = otpw_1y.select(['TAIL_NUM','FL_DATE','CANCELLED','DIVERTED','ARR_DELAY','ARR_DELAY_GROUP','ACTUAL_ELAPSED_TIME','ARR_DEL15','ARR_DELAY_NEW','AIR_TIME','WHEELS_ON','TAXI_IN', 'ARR_TIME','DEP_DELAY','DEP_DELAY_GROUP','DEP_DEL15','DEP_DELAY_NEW','DEP_TIME'])
funcs.count_missing_values(otpw_1y_select_cols_df, sort=True)

# dropping cxl obs still leaves some nulls in: ARR_DELAY,ARR_DELAY_GROUP,
# ACTUAL_ELAPSED_TIME,ARR_DEL15,ARR_DELAY_NEW,AIR_TIME,WHEELS_ON,TAXI_IN,ARR_TIME 	

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check diverted flights: retain

# COMMAND ----------

# check diverted flights 
otpw_1y_select_cols_df = otpw_1y.select(['TAIL_NUM','FL_DATE','CANCELLED','DIVERTED','ARR_DELAY','ARR_DELAY_GROUP','ACTUAL_ELAPSED_TIME','ARR_DEL15','ARR_DELAY_NEW','AIR_TIME','WHEELS_ON','TAXI_IN', 'ARR_TIME','DEP_DELAY','DEP_DELAY_GROUP','DEP_DEL15','DEP_DELAY_NEW','DEP_TIME']).filter(F.col('DIVERTED')==1)
funcs.count_missing_values(otpw_1y_select_cols_df, sort=True)

# COMMAND ----------

# MAGIC %md
# MAGIC So all of the remaining flights in the data with missing AIR_TIME, ARR_DELAY, ARR_DELAY_GROUP, ACTUAL_ELAPSED_TIME, ARR_DEL15, and ARR_DELAY_NEW values are diverted. I think these are OK to retain in the data for now until we know more about them. 
# MAGIC
# MAGIC Howevever, only about a small proportion (17%) of flights with missing `WHEELS ON`, `TAXI_IN`, and `ARR_TIME` data are diverted. Let's return to these in the suqsequent EDA steps. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop additional redundant/irrelevant features 

# COMMAND ----------

# list all string columns 
string_columns = [field.name for field in otpw_1y.schema.fields if isinstance(field.dataType, StringType)]
string_columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generate tables of origin/destination weather station information not used in modeling
# MAGIC - `origin_station_lon`/`origin_airport_lat` &  `dest_station_lat`/`dest_station_lon`: while the distance between the weather station and airport may be relevant to the precision of the weather measurements, this data is already reflected in `origin_station_dis`/`dest_station_dis` (retained in the data)
# MAGIC -  `origin_station_name`, `origin_iata_code`, `origin_region`,
# MAGIC

# COMMAND ----------

# generate look up tables of unecessary weather station features and write to storage

# list columns to generate lookup tables for origin stations
origin_stn_cols= ['origin_station_id','origin_station_name',
              'origin_iata_code','origin_region',
              'origin_station_lat','origin_station_lon'] 

origin_stn_lookup_df = funcs.make_lookup_table(otpw_1y, origin_stn_cols)
origin_stn_lookup_df.head()
funcs.write_pd_df_to_storage(origin_stn_lookup_df,"/LH/lookup_tables")

# repeat for destination stations (same info, but prefixes may be useful for later joins)
dest_stn_cols= ['dest_station_id','dest_station_name',
            'dest_iata_code','dest_region',
            'origin_station_lat','origin_station_lon'] 
dest_stn_lookup_df = funcs.make_lookup_table(otpw_1y, dest_stn_cols)
funcs.write_pd_df_to_storage(dest_stn_lookup_df,"/LH/lookup_tables")

# COMMAND ----------


# drop all of the above except for origin and destination station ids from the main df 
drop_stn_cols = ['origin_station_name','origin_iata_code','origin_region',
                 'origin_station_lat','origin_station_lon',
                 'dest_station_name','dest_iata_code','dest_region',
                 'dest_station_lat','dest_station_lon']

funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*drop_stn_cols)
funcs.get_df_dimensions(otpw_1y)              


# COMMAND ----------

# MAGIC %md
# MAGIC Columns that don't contribute useful information and can be dropped: 
# MAGIC - destination/origin IATA code: Code assigned by IATA and commonly used to identify a carrier. As the same code may have been assigned to different carriers over time, the code is not always unique. For analysis, use the Unique Carrier Code.
# MAGIC - destination/origin WAC: essentially a continent ID, far less granular than airport latitiude & longitude
# MAGIC - region: essentially state/country code
# MAGIC - station name:redundant given that we're retaining station ID
# MAGIC
# MAGIC Retain for now but may not be useful: 
# MAGIC - ICAO airport code: location indicator used by air traffic control and airline operations such as flight planning. ICAO codes are also used to identify other aviation facilities such as weather stations, international flight service stations or area control centers, whether or not they are located at airports. Flight information regions are also identified by a unique ICAO-code. 
# MAGIC - Origin Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time.
# MAGIC - Origin Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market.

# COMMAND ----------

# select all "origin" airport columns to determine which can be removed & added to look-up table
all_cols = otpw_1y.columns
flt_origin_id_cols = [col for col in all_cols if col.lower().startswith("origin_")]
flt_origin_id_cols

# COMMAND ----------

# create airport ID info tables & write to storage

origin_airport_id_cols = ['ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID',
                          'ORIGIN_CITY_MARKET_ID','ORIGIN_CITY_NAME',
                          'ORIGIN_STATE_ABR','ORIGIN_STATE_FIPS',
                          'ORIGIN_STATE_NM','ORIGIN_WAC',
                          'origin_airport_name','origin_icao',
                          'origin_type',
                          'origin_airport_lat','origin_airport_lon'] 

dest_airport_id_cols = ['DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID',
                          'DEST_CITY_MARKET_ID','DEST_CITY_NAME',
                          'DEST_STATE_ABR','DEST_STATE_FIPS',
                          'DEST_STATE_NM','DEST_WAC',
                          'dest_airport_name','dest_icao',
                          'dest_type',
                          'dest_airport_lat','dest_airport_lon']                        

origin_airport_lookup_df = funcs.make_lookup_table(otpw_1y, origin_airport_id_cols)
funcs.write_pd_df_to_storage(origin_airport_lookup_df,"/LH/lookup_tables")

dest_airport_lookup_df = funcs.make_lookup_table(otpw_1y, dest_airport_id_cols)
funcs.write_pd_df_to_storage(dest_airport_lookup_df,"/LH/lookup_tables")


# COMMAND ----------

# quick look 
origin_airport_lookup_df.head()

# COMMAND ----------

# view remaining columms 
otpw_1y.columns

# COMMAND ----------

# drop selected (non-useful) ID columns from main dataframe

drop_airport_id_cols = ['ORIGIN_CITY_NAME','DEST_CITY_NAME',
                         'ORIGIN_AIRPORT_SEQ_ID','DEST_AIRPORT_SEQ_ID',
                         'ORIGIN_STATE_FIPS','DEST_STATE_FIPS',
                         'ORIGIN_STATE_NM','DEST_STATE_NM',
                         'ORIGIN_WAC','DEST_WAC'] 

funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(* drop_airport_id_cols)
funcs.get_df_dimensions(otpw_1y)


# COMMAND ----------

# remaining columns 
otpw_1y.columns

# COMMAND ----------

# write to storage until next stage of EDA 
otpw_1y.write.mode('overwrite').parquet(f"{team_blob_url}/LH/1yr_clean_temp")

# COMMAND ----------


