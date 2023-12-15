# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Technical notebook to create engineered features

# COMMAND ----------

# importing custom functions
from Code.funcs import blob_connect, write_parquet_to_blob
import csv
import json
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, FloatType, DoubleType,  ArrayType, StringType
from pyspark.sql.functions import size, to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when, collect_list, udf, date_trunc, count, lag, first, last, percent_rank, array, year, month, dayofmonth
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, IndexToString, StringIndexerModel
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression as LR

team_blob_url = blob_connect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read from storage

# COMMAND ----------

# imputed, 2015-2019 dataset
imputed_data = spark.read.parquet(f"{team_blob_url}/BK/clean_5yr_WITHOUT_2019_imputed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### New engineered features

# COMMAND ----------

# MAGIC %md 
# MAGIC #### `day_of_year`
# MAGIC - description: day of year out of 366
# MAGIC - value range: 1-366
# MAGIC - dtype: int
# MAGIC

# COMMAND ----------

# use pyspark built in function for day of year
imputed_data = imputed_data.withColumn(
    "day_of_year",
    F.dayofyear(F.col("calendar_date"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### `isHolidayWindow`
# MAGIC - description: 1 if the calendar_date falls within 2 days before / 2 days after a holiday, 0 otherwise
# MAGIC - value range: Binary (0, 1)
# MAGIC - dtype: Integer

# COMMAND ----------

# list generated from ChatGPT
# federal holidays (excludes bank holidays or things like Halloween)
holidays_list = [
    "2015-01-01", "2015-01-19", "2015-02-16", "2015-05-25", "2015-07-04", "2015-09-07", "2015-10-12", "2015-11-11", "2015-11-26", "2015-12-25",
    "2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04", "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-25", "2016-12-26",
    "2017-01-01", "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04", "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-11", "2017-11-23", "2017-12-25",
    "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04", "2018-09-03", "2018-10-08", "2018-11-11", "2018-11-12", "2018-11-22", "2018-12-25",
    "2019-01-01", "2019-01-21", "2019-02-18", "2019-05-27", "2019-07-04", "2019-09-02", "2019-10-14", "2019-11-11", "2019-11-28", "2019-12-25"
]

extended_holidays_list = []

# go through each holiday date and generate dates 2 days before and 2 days after
for holiday in holidays_list:
    holiday_date = datetime.strptime(holiday, "%Y-%m-%d")
    for i in range(-2, 3):
        extended_date = (holiday_date + timedelta(days=i)).strftime("%Y-%m-%d")
        extended_holidays_list.append(extended_date)

# remove duplicates and sort
extended_holidays_list = sorted(list(set(extended_holidays_list)))
holidays_set = set(extended_holidays_list)

# broadcast
broadcast_holidays_set = spark.sparkContext.broadcast(holidays_set)

# Function to check if a date falls within the extended holiday window
def is_holiday_window_extended(date):
    return 1 if date in extended_holidays_list else 0

# Register the function as a UDF
is_holiday_window_extended_udf = udf(is_holiday_window_extended)

# Add the 'isHolidayWindow' column with extended holiday window
imputed_data = imputed_data.withColumn('isHolidayWindow',
    is_holiday_window_extended_udf(col('calendar_date'))
)

# COMMAND ----------

# DBTITLE 1,Check Bailey Features
# verify -- can coment out after
test_date = "2017-07-05"  #day after 4th of july
filtered_df = imputed_data.filter(col('calendar_date') == test_date).select('calendar_date', 'day_of_year', 'isHolidayWindow')
filtered_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### `Av_airport_delay`
# MAGIC - description: Average departure delay at the origin. Averaged over 4 hour windew from 6 to 2 hour prior to scheduled departure
# MAGIC - value range: -infinity: + infinity
# MAGIC - dtype: double
# MAGIC
# MAGIC #### `Av_carrier_delay`
# MAGIC - description: Average departure delay of the airline. Averaged over 4 hour windew from 6 to 2 hour prior to scheduled departure
# MAGIC - value range: -infinity: + infinity
# MAGIC - dtype: double
# MAGIC
# MAGIC #### `Prev_delay_tailnum`
# MAGIC - description: Last known departure delay for the aircarf of the flight in question. Determined for 24 hour window from 26 to 2 hour prior to the scheduled departure. 0 if the aircraft did not record any flights in that window.
# MAGIC - value range: -infinity: + infinity
# MAGIC - dtype: double
# MAGIC
# MAGIC #### `weekly_flights_tailnum`
# MAGIC - description: Total number of flights the aircarf logged over the most recent week. Determined for 168 hour window from 170 to 2 hour prior to the scheduled departure. 0 if the aircraft did not record any flights in that window.
# MAGIC - value range: -infinity: + infinity
# MAGIC - dtype: integer
# MAGIC
# MAGIC #### `hourly_flights_origin`
# MAGIC - description: Number of flights scheduled to depart from the origin airport with 1 hour prior to scheduled departure of the flight in question. Relies on the assumption that the flight schedule is already avaliable 2 hours prior to departure. 
# MAGIC - value range: -infinity: + infinity
# MAGIC - dtype: double
# MAGIC
# MAGIC #### `airport_average_hourly`
# MAGIC - description: Average number of flights departing from the origin airport every hour. Averaged over a week, but with a correaction factor to take into account the fact that during the night there are very few flights. To correct for that, we empirically determined that there are approximately 100 hours every week when most flights depart. We use this number for averaging, rahter than astronomical 168 hours. 
# MAGIC - value range: 0: + infinity
# MAGIC - dtype: double
# MAGIC
# MAGIC #### `precip_severity`
# MAGIC - description: Precepitation averaged over 3 hours. To introduce non-linearity, we zero all values below 0.3, this dropping anything that is less that moderate rain.
# MAGIC - value range: 0: + infinity
# MAGIC - dtype: double
# MAGIC
# MAGIC #### `snow_severity`
# MAGIC - description: Snow accumulation over previous day. To introduce non-linearity, we zero all values below 3, this dropping anything that is less that moderate snow.
# MAGIC - value range: 0: + infinity
# MAGIC - dtype: double

# COMMAND ----------

# Make new column with time in seconds since the begining of Unix epoch
df_clean = imputed_data.withColumn('time_long', imputed_data.sched_depart_date_time_UTC.cast("long")).orderBy(imputed_data.sched_depart_date_time_UTC)

#Helper function to navigate this column
hours = lambda i: i * 3600

# previos delay for this tail number
Tail_Window_1d = Window.partitionBy('TAIL_NUM').orderBy(col('time_long')).rangeBetween(-hours(26), -hours(2))
Tail_Window_1wk = Window.partitionBy('TAIL_NUM').orderBy(col('time_long')).rangeBetween(-hours(170), -hours(2))
# average delay for this airline
Carrier_Window_4h = Window.partitionBy('OP_UNIQUE_CARRIER').orderBy(col('time_long')).rangeBetween(-hours(6), -hours(2))
# average delay for this airport
Origin_Window_4h = Window.partitionBy('ORIGIN').orderBy(col('time_long')).rangeBetween(-hours(6), -hours(2))
Origin_Window_1h = Window.partitionBy('ORIGIN').orderBy(col('time_long')).rangeBetween(-hours(1), -hours(0))
Origin_Window_1wk = Window.partitionBy('ORIGIN').orderBy(col('time_long')).rangeBetween(-hours(168), -hours(0))

df_clean = df_clean.withColumns({
    "Av_airport_delay": _mean("DEP_DELAY").over(Origin_Window_4h)
    ,"Prev_delay_tailnum": last("DEP_DELAY").over(Tail_Window_1d)
    ,"Av_carrier_delay": _mean("DEP_DELAY").over(Carrier_Window_4h)
    ,"weekly_flights_tailnum": count(col("DEP_DELAY")).over(Tail_Window_1wk)
    ,"hourly_flights_origin": count(col("DEP_DELAY")).over(Origin_Window_1h)
    ,'airport_average_hourly': count(col("DEP_DELAY")).over(Origin_Window_1wk)/100
    ,'precip_severity': when(df_clean.origin_3Hr_Precipitation >= 0.3,df_clean.origin_3Hr_Precipitation).otherwise(0) 
    ,'snow_severity': when(df_clean.origin_DailySnowfall >= 3,df_clean.origin_DailySnowfall).otherwise(0)
    }).fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### `airport_congestion`
# MAGIC - description: Ratio between the number of flights within an hour before departure and the average hourly number of flights at the airport. To make this metric of airport congestion more robust we zero if there are too few flights this hour, or too few flights on average. 
# MAGIC - value range: 0: + infinity
# MAGIC - dtype: double

# COMMAND ----------

df_clean = df_clean.withColumns({
    'airport_congestion': when(
                            (col("hourly_flights_origin") >= 2) \
                            & (col('airport_average_hourly') >= 2) \
                            , col("hourly_flights_origin")/col('airport_average_hourly')) \
                            .otherwise(0)
    }).fillna(0)

# COMMAND ----------

gre_histogram = df_clean.select('airport_congestion').rdd.flatMap(lambda x: x).histogram(100)

# Loading the Computed Histogram into a Pandas Dataframe for plotting
pd.DataFrame(
    list(zip(*gre_histogram)), 
    columns=['bin', 'frequency']
).set_index(
    'bin'
).plot(kind='bar');

# COMMAND ----------

gre_histogram

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write combined dataset to blob

# COMMAND ----------

location = 'BK/clean_5yr_WITHOUT_2019_eng'
write_parquet_to_blob(df_clean, location)

# COMMAND ----------

#read them back from blob
df_clean = spark.read.parquet(f"{team_blob_url}/{location}")

# COMMAND ----------


