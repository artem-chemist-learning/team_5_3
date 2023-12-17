# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 EDA: Re-joined 1-year OTPW Dataset
# MAGIC
# MAGIC **Team 5-3: Bailey Kuehl, Lucy Herr, Artem Lebedev, Erik Sambraillo** 
# MAGIC <br>**261 Fall 2023**
# MAGIC <br>
# MAGIC <br>
# MAGIC This notebook (03) is the third step of the full EDA process for the 1-year OTPW dataset (2015). This version of the data has been re-joined from the raw flights and weather tables in order to retain a higher proportion of the original weather feature values.
# MAGIC
# MAGIC Overview of EDA & data cleaning steps performed in this notebook:
# MAGIC 1. time/date feature selection
# MAGIC 2. delay time features review
# MAGIC 3. arrival time features review
# MAGIC 4. remaining string features 
# MAGIC 3. address/impute remaining nulls 

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
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.stat import Correlation
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

# connect to team blob
team_blob_url = funcs.blob_connect()
# view blob storage root folder 
display(dbutils.fs.ls(f"{team_blob_url}/"))

# COMMAND ----------

# load re-joined 1-year OTPW data with corrected schema 
otpw_1y = spark.read.parquet(f"{team_blob_url}/LH/1yr_clean_temp")
# view dimensions of 1-year joined data
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

otpw_1y.columns

# COMMAND ----------

display(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review delay & arrival features 
# MAGIC
# MAGIC | Feature Name | Type | Description | Next Steps |
# MAGIC | -------- | ------- | -------- | -------- |
# MAGIC |**`DEP_DELAY`**|  metric (minutes) | predicted outcome (metric) -  Difference in minutes between scheduled and actual departure time. Early departures show negative numbers. |predicted continuous outcome, need to address class imbalance |
# MAGIC |**`DEP_DEL15`**| indicator | Departure Delay Indicator, 15 Minutes or More (1=Yes)| predicted binary outcome for log. reg., need to address class imbalance | 
# MAGIC |`TAXI_OUT`| metric (minutes)| taxi-out time in minutes | consider as portion of flight/delay time |
# MAGIC |`TAXI_IN`| metric (minutes) | taxi-in time in minutes | consider as portion of flight/delay time | 
# MAGIC |`ARR_DELAY`| metric (minutes) | Difference in minutes between scheduled and actual arrival time. Early arrivals show negative numbers.| FE potential: previous arr delay at origin of incoming flights, arr delay at destination for arriving flights, can we break down at flight-number level? | 
# MAGIC |`DIVERTED` | indicator | some nulls in relevant features for these obs. - retain  | could use as separate class in multi-class pred |
# MAGIC |`DISTANCE` | metric (miles) | distance between origin & dest. airport (miles) ||
# MAGIC
# MAGIC
# MAGIC | Feature Name | Type | Description | Next Steps |
# MAGIC | -------- | ------- | -------- | -------- |
# MAGIC |`ARR_DEL15` | indicator | Arrival Delay Indicator, 15 Minutes or mode | maybe useful for log reg? |
# MAGIC |`DEP_DELAY_GROUP`| interval | 15-min. Departure Delay intervals (<-15 to >180) | | 
# MAGIC |`ARR_DELAY_GROUP`| interval |15-min. Arrival Delay intervals (<-15 to >180) ||
# MAGIC |`DEP_TIME_BLK`| interval | CRS Dep. Time Block, Hourly Intervals (scheduled) ||
# MAGIC |`ARR_TIME_BLK`| interval | CRS Departure Arr. time Block, Hourly Intervals (scheduled) ||
# MAGIC |`WHEELS_OFF`| local time: hhmm | time of take-off | drop - w/ `WHEELS_ON`, redundant to airtime |
# MAGIC |`WHEELS_ON`|local time: hhmm | time of take-off | drop - w/ `WHEELS_OFF`, redundant to airtime |
# MAGIC | `CRSElapsedTime`	| metric (minutes) | CRS Elapsed Time of Flight = scheduled flight duration| | 
# MAGIC | `ActualElapsedTime`	| metric (minutes) | Elapsed Time of Flight = actual flight duration|  EDA Q: consider in-air delays pre-landing pileups? |  
# MAGIC | `AirTime`	| metric (minutes)| Flight Time, in Minutes	|  how is this calculated? |
# MAGIC
# MAGIC | Feature Name | Type | Description | Next Steps |
# MAGIC | -------- | ------- | -------- | -------- |
# MAGIC |`FLIGHTS`| 1 | each flight observation row = 1  | drop since this doesn't contribute information for prediction |
# MAGIC |`CANCELLED`| indicator | Cancelled flights = 0 |drop since all obs should =0 after removing cancelled flights|
# MAGIC |`DISTANCE_GROUP`| interval | drop (less granular than distance) | |
# MAGIC |`ARR_DELAY_NEW`| metric | omit - less granular than `ARR_DELAY`|Difference in minutes between scheduled and actual arrival time, early arrivals = 0 |
# MAGIC |`DEP_DELAY_NEW`| metric| Difference in minutes between scheduled and actual arrival time, early arrivals = 0 |omit - less granular|
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### `DEP_DELAY`, `DEP_DELAY_NEW`: compare
# MAGIC - `DEP_DELAY`: Difference in minutes between scheduled and actual departure time. Early departures show negative numbers. 
# MAGIC - `DEP_DELAY_NEW` (`DepDelayMinutes` in documentation): Difference in minutes between scheduled and actual departure time. Early departures set to 0.

# COMMAND ----------

# compare DEP_DELAY vs. DEP_DELAY_NEW: what percentage of values differ across columns?
funcs.calculate_percentage_difference(otpw_1y, 'DEP_DELAY', 'DEP_DELAY_NEW')

# COMMAND ----------

delay_feats = ['DEP_DELAY', 'DEP_DELAY_NEW']
funcs.pairplot(otpw_1y, delay_feats, sample_size=10000)


# COMMAND ----------

# what % of dep_delay flights are labeled as early (negative minutes value)? 
average_early_time = otpw_1y.filter(F.col('DEP_DELAY') < 0).agg(F.avg('DEP_DELAY')).first()[0]
print(f"Average Minutes Early for Early flights: {average_early_time}")

# COMMAND ----------

# MAGIC %md
# MAGIC Interesting - so although enough flights are "early" to result in a major discrepancy between the two columns (e.g., negative in DEP_DELAY but '0' in DEP_DELAY_NEW), these flights tend to be only a few minutes early on average. Given that DEP_DELAY represents a more precise measurement of this trend, this feature makes more sense to use as the metric target for prediction. 

# COMMAND ----------

funcs.plot_boxplots(otpw_1y, delay_feats, sample_fraction=0.1)

# COMMAND ----------

# drop dep_delay_new so that we can use dep_delay (more granular measure of delay time)
funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop('DEP_DELAY_NEW')
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

otpw_1y.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### `DEP_DEL15`, `DEP_DEL_BLK`, `DEP_TIME_GRP`

# COMMAND ----------

# what about 'DEP_DEL15'?
DEP_DEL15_counts_df = otpw_1y.groupBy('DEP_DEL15').count()
#sort descending
DEP_DEL15_counts_df = DEP_DEL15_counts_df.orderBy(F.desc('count'))
DEP_DEL15_counts_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC `DEP_DEL15` can be used in any models predicting delay as a discrete class, so we'll retain it to be used when the continous `DEP_DELAY` outcome isn't applicable.<br><br>
# MAGIC **Important**: based on the much-higher proportion of non-delayed flights, we need to implement down-sampling or another technique to address class imbalance in our models. 

# COMMAND ----------

#what about'DEP_DELAY_GROUP' and 'DEP_TIME_BLK'? 
DEP_DELAY_GROUP_counts_df = otpw_1y.groupBy('DEP_DELAY_GROUP').count()

DEP_DELAY_GROUP_counts_df = DEP_DELAY_GROUP_counts_df.orderBy(F.asc('DEP_DELAY_GROUP'))
DEP_DELAY_GROUP_counts_df = DEP_DELAY_GROUP_counts_df.toPandas()
DEP_DELAY_GROUP_counts_df['New_String_Column'] = (DEP_DELAY_GROUP_counts_df['DEP_DELAY_GROUP'] * 15).astype(str) + ' min'
DEP_DELAY_GROUP_counts_df

# COMMAND ----------


from matplotlib.ticker import FuncFormatter

# plot dep delay group 
plt.bar(DEP_DELAY_GROUP_counts_df['New_String_Column'],height=DEP_DELAY_GROUP_counts_df["count"])
plt.xticks(rotation=45)

# Define a formatter function
def millions_formatter(x, pos):
    return f'{int(x)}'

# Create a FuncFormatter object using the custom formatter function
formatter = FuncFormatter(millions_formatter)

# Get current axis
ax = plt.gca()

# Set the formatter for the y-axis
ax.yaxis.set_major_formatter(formatter)
plt.xlabel('Dep. Delay Group (15-minute intervals)')
plt.ylabel('Number of Flights')
plt.title('Total Flights by Departure Delay Group (2015)')

# Show the plot
plt.show()

# COMMAND ----------

# plot dep time block
DEP_TIME_BLK_counts_df = otpw_1y.groupBy('DEP_TIME_BLK').count()

# DEP_DELAY_GROUP_counts_df = DEP_DELAY_GROUP_counts_df.orderBy(F.asc('DEP_DELAY_GROUP'))
# DEP_DELAY_GROUP_counts_df = DEP_DELAY_GROUP_counts_df.toPandas()
# DEP_DELAY_GROUP_counts_df['New_String_Column'] = (DEP_DELAY_GROUP_counts_df['DEP_DELAY_GROUP'] * 15).astype(str) + ' min'
# DEP_DELAY_GROUP_counts_df 
# plt.bar(DEP_TIME_BLK_counts_df['New_String_Column'],height=DEP_DELAY_GROUP_counts_df["count"])
# plt.xticks(rotation=45)

# plt.xlabel('Dep. Delay Group (15-minute intervals)')
# plt.ylabel('Number of Flights')
# plt.title('Total Flights by Departure Delay Group (2015)')

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can safely drop `DEP_TIME_BLK` from our modeling data without losing information. `DEP_DELAY_GROUP` is also less granular than `DEP_DELAY`. However, it may be useful to feature engineer some measure of flight volume delay volume per short interval in order to examine scenarios where earlier delays (at or before 2-hours prior to departure) propagate to later delays `DEP_DELAY_GROUP`. 

# COMMAND ----------

# DROP 'DEP_DELAY_GROUP', 'DEP_TIME_BLK'
extra_del_cols = ['DEP_DELAY_GROUP', 'DEP_TIME_BLK']
funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*extra_del_cols)
funcs.get_df_dimensions(otpw_1y)


# COMMAND ----------

# MAGIC %md
# MAGIC ### `ARR_DELAY`, `ARR_DELAY_NEW`, & `ARR_TIME_BLK`: compare 

# COMMAND ----------


arr_feats = ['ARR_DELAY','ARR_DELAY_NEW']
funcs.plot_boxplots(otpw_1y, arr_feats, sample_fraction=0.1)

# COMMAND ----------

# MAGIC %md
# MAGIC So we see the same relationship between `ARR_DELAY` and `ARR_DELAY_NEW`, and can remove `ARR_DELAY_NEW` from the data. 

# COMMAND ----------

# hourly arr time block
ARR_TIME_BLK_counts_df = otpw_1y.groupBy('ARR_TIME_BLK').count()
display(ARR_TIME_BLK_counts_df)

# COMMAND ----------

# plot arrival delay group 
ARR_DELAY_GROUP_counts_df = otpw_1y.groupBy('ARR_DELAY_GROUP').count()

ARR_DELAY_GROUP_counts_df = ARR_DELAY_GROUP_counts_df.orderBy(F.asc('ARR_DELAY_GROUP'))
ARR_DELAY_GROUP_counts_df = ARR_DELAY_GROUP_counts_df.toPandas()
ARR_DELAY_GROUP_counts_df['New_String_Column'] = (ARR_DELAY_GROUP_counts_df['ARR_DELAY_GROUP'] * 15).astype(str) + ' min'

plt.bar(ARR_DELAY_GROUP_counts_df['New_String_Column'],height=ARR_DELAY_GROUP_counts_df["count"])
plt.xticks(rotation=45)

# Define a formatter function
def millions_formatter(x, pos):
    return f'{int(x)}'

# Create a FuncFormatter object using the custom formatter function
formatter = FuncFormatter(millions_formatter)

# Get current axis
ax = plt.gca()

# Set the formatter for the y-axis
ax.yaxis.set_major_formatter(formatter)
plt.xlabel('Arr. Delay Group (15-minute intervals)')
plt.ylabel('Number of Flights')
plt.title('Total Flights by Arrival Delay Group (2015)')

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The arrival delay groups appear to follow a similar distribution to the departure delay groups. Definitely something to consider in subsewuent FE. In the meantime, we can drop these 3 extraneous arrival columns from the data. 

# COMMAND ----------

# DROP extra delay cols 
extra_arr_cols = ['ARR_DELAY_NEW', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK']
funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*extra_arr_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### `FLIGHTS` & `CANCELLED`: Check & drop

# COMMAND ----------

# confirm that we can drop "flights" features (all values == 1)
distinct_flights_vals = otpw_1y.select('FLIGHTS').distinct()
display(distinct_flights_vals)

# COMMAND ----------

# confirm that we can drop "cancelled" feature (all values == 0)
distinct_cancelled_vals = otpw_1y.select('CANCELLED').distinct()
display(distinct_cancelled_vals)

# COMMAND ----------

# drop 'FLIGHTS' and 'CANCELLED' columns 
drop_cols = ['FLIGHTS','CANCELLED']
funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*drop_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### `TAXI_OUT`,`DEP_TIME`,`WHEELS_OFF`: compare

# COMMAND ----------

taxi_out_feats = ['TAXI_OUT','DEP_TIME','WHEELS_OFF']
funcs.plot_boxplots(otpw_1y,taxi_out_feats)

# COMMAND ----------

funcs.histogram(otpw_1y, 'TAXI_OUT')

# COMMAND ----------

funcs.histogram(otpw_1y, 'TAXI_IN')

# COMMAND ----------

taxi_feats = ['TAXI_IN','TAXI_OUT','DEP_DELAY']
funcs.pairplot(otpw_1y, taxi_feats, sample_size=3000)

# COMMAND ----------

# MAGIC %md
# MAGIC ### `TAXI_OUT`/`TAXI_IN` vs. `DEP_TIME`/`ARR_TIME` - `WHEELS_OFF`/`WHEELS_ON`: 
# MAGIC compare for reduncy & drop `WHEELS_OFF`/`WHEELS_ON`

# COMMAND ----------

# check that taxi out time (minutes)= 'WHEELS_OFF' (local time hhmm) - actual departure time (local hhmm)
# select relevant cols only for efficiency 
wheelsoff_taxi_comp_df = otpw_1y.select('TAXI_OUT','WHEELS_OFF','DEP_TIME') 

#calc time difference colunn (wheels off - dep time)
wheelsoff_taxi_comp_df = wheelsoff_taxi_comp_df\
    .withColumn("wheelsoff_min_deptime",funcs.calculate_time_difference(F.col("WHEELS_OFF"), F.col("DEP_TIME")))

# filter rows where wheels off - dep time != taxi out time 
wheelsoff_taxi_comp_df = wheelsoff_taxi_comp_df\
    .withColumn("taxi_out_check", F.col("wheelsoff_min_deptime") == F.col("TAXI_OUT"))

wheelsoff_taxi_comp_df_f = wheelsoff_taxi_comp_df.filter(F.col('taxi_out_check')==False)
display(wheelsoff_taxi_comp_df_f)


# COMMAND ----------

# check that taxi in time (minutes)= 'WHEELS_ON' (local time hhmm) - actual arrival time (local hhmm)
# select relevant cols only for efficiency 
wheelson_taxiin_comp_df = otpw_1y.select('TAXI_IN','WHEELS_ON','ARR_TIME') 

#calc time difference colunn (wheels off - dep time)
wheelson_taxiin_comp_df = wheelson_taxiin_comp_df\
    .withColumn("arrtime_min_wheelson",funcs.calculate_time_difference(F.col("ARR_TIME"), F.col("WHEELS_ON")))

# filter rows where wheels off - dep time != taxi out time 
wheelson_taxiin_comp_df = wheelson_taxiin_comp_df\
    .withColumn("taxi_in_check", F.col("arrtime_min_wheelson") == F.col("TAXI_IN"))

wheelson_taxiin_comp_df_f = wheelson_taxiin_comp_df.filter(F.col('taxi_in_check')==False)
display(wheelson_taxiin_comp_df_f)

# COMMAND ----------

# MAGIC %md
# MAGIC Because we've confirmed that all `TAXI_IN` and `TAXI_OUT` times represent the difference between `ARR_TIME` and `WHEELS_ON` or `DEP_TIME` and `WHEELS_OFF` (respectively), we can safely drop the `WHEELS_OFF` and `WHEELS_ON` columns without losing information. 

# COMMAND ----------

# drop 'WHEELS_OFF' and 'WHEELS_ON' columns 
drop_cols = ['WHEELS_OFF','WHEELS_ON']
funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*drop_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC next steps: 
# MAGIC -  check  'CRS_ARR_TIME' redundant to ARR_DELAY' + 'ARR_TIME' ? 
# MAGIC -  can drop `ARR_DEL_15 too
# MAGIC -  'CRS_ELAPSED_TIME' = CRS_ARR_TIME - CRS_ARR_TIME ? 
# MAGIC -  'ACTUAL_ELAPSED_TIME' = ARR_TIME - DEP_TIME ? 
# MAGIC -  FL date redundant to schedeled dep date utc? 
# MAGIC
# MAGIC general EDA checks:
# MAGIC -  don't need 'DISTANCE_GROUP' but can use to overview trends before droppng
# MAGIC -  broad overview: 'ORIGIN_CITY_MARKET_ID',' 'DEST_CITY_MARKET_ID',
# MAGIC -  broad overview: by state (state name) 
# MAGIC - 'OP_UNIQUE_CARRIER' vs. 'OP_CARRIER_AIRLINE_ID' / 'OP_CARRIER'  --> cross check for discrepancies
# MAGIC - ICAO
# MAGIC - STATION ID
# MAGIC - 'DEST' vs. DEST AIRPORT ID, 'ORIGIN; vs. ORIGIN AIRPORT ID,
# MAGIC -  weather: sort out which features can be used for immediate prediction, vs. which have to be feature engineered 

# COMMAND ----------

# MAGIC %md
# MAGIC ### `CRS_ARR_TIME`: redundant to `ARR_DELAY` + `ARR_TIME`?

# COMMAND ----------

# check if crs elapsed time - actual elapsed time = dep_delay 
crs_arr_time_comp_df = otpw_1y.select('ARR_TIME','ARR_DELAY','CRS_ARR_TIME')

def calculate_time_difference(end_time_col, start_time_col):
    """
    Calculates the time difference in minutes for columns in numeric HHMM or HMM format, accounting for day changes.
    """
    # extract hours and minutes from the float format
    end_time_col_hours = F.floor(end_time_col / 100)
    end_time_col_minutes = end_time_col % 100
    start_time_col_hours = F.floor(start_time_col / 100)
    start_time_col_minutes = start_time_col % 100
    # convert to total minutes
    end_time_col_total = (end_time_col_hours * 60) + end_time_col_minutes
    start_time_col_total = (start_time_col_hours * 60) + start_time_col_minutes
    # check for day change (end time up to 2am but start time after 8pm)
    condition = (end_time_col_hours >=0) & (end_time_col_hours < 3) & (start_time_col_hours > 18)
    # add 24 hours (1440 minutes) to end_time_col_total if day change
    return F.when(condition, end_time_col_total + 1440).otherwise(end_time_col_total) - start_time_col_total

#calc time difference colunn (wheels off - dep time)
crs_arr_time_comp_df = crs_arr_time_comp_df\
    .withColumn("crsarrtime_min_actual",calculate_time_difference(F.col("CRS_ARR_TIME"),F.col("ARR_TIME")))

# filter rows where wheels off - dep time != taxi out time 
crs_arr_time_comp_df = crs_arr_time_comp_df\
    .withColumn("arr_time_check", F.col("crsarrtime_min_actual") == F.col("ARR_DELAY"))

crs_arr_time_comp_df_f = crs_arr_time_comp_df.filter(F.col('arr_time_check')==False)
display(crs_arr_time_comp_df_f)

# COMMAND ----------

# MAGIC %md
# MAGIC Disregarding the reversal of the sign from calculation the comparison value, these results seem to confirm that `CRS_ARR_TIME` + `ARR_DELAY` = (actual) `ARR_TIME`.  

# COMMAND ----------

# MAGIC %md 
# MAGIC **TO DO: follow this up** --> some of these mismatches due to how the comparison function is written, but it appears that a meaningful proportion of the ARR_TIME & CRS_ARR_TIME values don't adhere to the HHMM format of the columns. 
# MAGIC
# MAGIC Similar checks:
# MAGIC - `CRS_ELAPSED_TIME` = `CRS_ARR_TIME` - `CRS_ARR_TIME`?
# MAGIC - `ACTUAL_ELAPSED_TIME` = `ARR_TIME` - `DEP_TIME` ?

# COMMAND ----------

# MAGIC %md
# MAGIC ### `DIVERTED`: check subset of diverted flights 

# COMMAND ----------

# check diverted flights subset
diverted_df = otpw_1y.filter(otpw_1y['DIVERTED'] == 1)
print(f"Percentage of diverted flight obs. in OTPW_1YR: {(diverted_df.count()/otpw_1y.count())*100}")
diverted_summary = diverted_df.describe()
diverted_summary.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC When flights are diverted, they do not have arrival delay (`ARR_DELAY`) or flight time information recorded (`AIR_TIME`). They may also be missing data for `WHEELS_ON`, `TAXI_IN`, etc. Given that these observations constitute a substantial proportion of our data, we need to retain these while we evaluate their potential impacts on our models. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select date/time features 
# MAGIC
# MAGIC | Feature Name:type | Description | Next Steps |
# MAGIC | -------- | ------- | -------- |
# MAGIC | `QUARTER` | retain for reference/partitioning  |  |
# MAGIC | `YEAR` | retain for reference/partitioning  | |
# MAGIC | **`DAY_OF_MONTH`** | convert to day of year to capture seasonality + month | |
# MAGIC | `MONTH`| can drop after doing above  | |
# MAGIC | `DAY_OF_WEEK`| examine trends in next stage of EDA | |
# MAGIC | `FL_DATE`| compare to other date features - choose 1 | |
# MAGIC | `sched_depart_date_time_UTC` | retain for FE | |
# MAGIC | `two_hours_prior_depart_UTC` |  retain for FE  | |
# MAGIC | `four_hours_prior_depart_UTC` | retain for FE | |
# MAGIC | `three_hours_prior_depart_UTC` | DROP | |
# MAGIC | `origin_DATE` | DROP | | |
# MAGIC | `origin_UTC` | DROP | |
# MAGIC | `dest_DATE` | DROP | |
# MAGIC | `dest_UTC` | DROP | |
# MAGIC
# MAGIC - `CRS_DEP_TIME`
# MAGIC - `DEP_TIME`
# MAGIC - `CRS_ARR_TIME`
# MAGIC - `ARR_TIME`
# MAGIC - `CRS_ELAPSED_TIME`
# MAGIC - `ACTUAL_ELAPSED_TIME`
# MAGIC - `AIR_TIME`

# COMMAND ----------

# drop additional unecessary/redundant features (left over from method for re-joining the flights and weather data)
# plot w/ delay, corr 
drop_date_time_cols = ['three_hours_prior_depart_UTC','four_hours_prior_depart_UTC',
                       'origin_DATE','origin_UTC','dest_DATE','dest_UTC']

funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*drop_date_time_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# add 'day_of_year' column
otpw_1y = otpw_1y.withColumn('day_of_year', F.dayofyear(F.col('FL_DATE')))

# reorder date columns for clarity
funcs.get_df_dimensions(otpw_1y)

reorder_cols = ['sched_depart_date_time_UTC','two_hours_prior_depart_UTC','DAY_OF_WEEK','day_of_year','MONTH']
all_cols = otpw_1y.columns
remaining_cols = [c for c in all_cols if c not in reorder_cols]
full_col_reorder_list = reorder_cols+remaining_cols
otpw_1y = otpw_1y.select(full_col_reorder_list)

funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check/impute any remaining null values 

# COMMAND ----------

nb03_missing_df = funcs.count_missing_values(otpw_1y)

# COMMAND ----------

funcs.filter_df_by_min_max(nb03_missing_df, 'null_percent', 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Impute Precipitation feature nulls as 0 (no precipitation in that time period)

# COMMAND ----------

# impute null values in 'dest_DailySnowfall' & 'origin_DailySnowfall' with 0
precip_impute_cols = ['origin_HourlyPrecipitation','dest_HourlyPrecipitation','origin_3Hr_Precipitation','dest_3Hr_Precipitation','origin_6Hr_Precipitation','dest_6Hr_Precipitation','origin_12Hr_Precipitation','dest_12Hr_Precipitation','origin_DailyPrecipitation','dest_DailyPrecipitation']

def impute_nulls_with_zero(df: DataFrame, columns: list) -> DataFrame:
    """
    Impute null values with zero for specific columns in a Spark DataFrame.
    Args:
    df (DataFrame): The input DataFrame.
    columns (list): List of column names to impute nulls as zero.
    Returns:
    DataFrame: The DataFrame with nulls imputed as zero in specified columns.
    """
    # Create a dictionary with column names and the value 0
    fill_values = {col: 0 for col in columns}
    # Impute nulls with zero
    df_imputed = df.fillna(fill_values)
    return df_imputed

print(f"Number of nulls in precip_impute_cols before imputing:")
for col in precip_impute_cols:
    print(f"{col}:{otpw_1y.filter(F.col(col).isNull()).count()}\n")

otpw_1y = impute_nulls_with_zero(otpw_1y, precip_impute_cols)

# COMMAND ----------

print(f"Number of nulls in precip_impute_cols after imputing:")
for col in precip_impute_cols:
    print(f"{col}:{otpw_1y.filter(F.col(col).isNull()).count()}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Impute DailyDepartureFromNormalAverageTemperature feature nulls as 0 (no daily departure from average)

# COMMAND ----------

# Impute DailyDepartureFromNormalAverageTemperature nulls with 0
dailydeptemp_impute_cols = ['origin_DailyDepartureFromNormalAverageTemperature','dest_DailyDepartureFromNormalAverageTemperature']

# print(f"Number of nulls in dailydeptemp_impute_cols before imputing:")
# for col in dailydeptemp_impute_cols:
#     print(f"{col}:{otpw_1y.filter(F.col(col).isNull()).count()}\n")

otpw_1y = impute_nulls_with_zero(otpw_1y, dailydeptemp_impute_cols)

# COMMAND ----------

print(f"Number of nulls in dailydeptemp_impute_cols after imputing:")
for col in dailydeptemp_impute_cols:
    print(f"{col}:{otpw_1y.filter(F.col(col).isNull()).count()}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Impute PressureChange feature nulls as 0 (no PressureChange in that time period)
# MAGIC

# COMMAND ----------

# Impute DailyDepartureFromNormalAverageTemperature nulls with 0
PressureChange_impute_cols = ['origin_3Hr_PressureChange','dest_3Hr_PressureChange','origin_6Hr_PressureChange','dest_6Hr_PressureChange','origin_12Hr_PressureChange','dest_12Hr_PressureChange']

# print(f"Number of nulls in PressureChange_impute_cols before imputing:")
# for col in PressureChange_impute_cols:
#     print(f"{col}:{otpw_1y.filter(F.col(col).isNull()).count()}\n")

otpw_1y = impute_nulls_with_zero(otpw_1y, PressureChange_impute_cols)


# COMMAND ----------

print(f"Number of nulls in PressureChange_impute_cols after imputing:")
for col in PressureChange_impute_cols:
    print(f"{col}:{otpw_1y.filter(F.col(col).isNull()).count()}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop missing values: 
# MAGIC - HourlyDryBulbTemperature
# MAGIC - DailyAverageDryBulbTemperature
# MAGIC - DailyAverageRelativeHumidity
# MAGIC - HourlyStationPressure, StationPressure, DailyAverageStationPressure 

# COMMAND ----------

funcs.get_df_dimensions(otpw_1y)

weather_feats_drop_nulls = ['origin_HourlyDryBulbTemperature','dest_HourlyDryBulbTemperature',
                            'origin_3Hr_DryBulbTemperature','dest_3Hr_DryBulbTemperature',
                            'origin_6Hr_DryBulbTemperature','dest_6Hr_DryBulbTemperature',
                            'dest_6Hr_DryBulbTemperature',
                            'origin_12Hr_DryBulbTemperature',
                            'dest_12Hr_DryBulbTemperature',
                            'origin_DailyAverageDryBulbTemperature','dest_DailyAverageDryBulbTemperature',
                            'origin_DailyAverageRelativeHumidity','dest_DailyAverageRelativeHumidity',
                            'origin_HourlyStationPressure','dest_HourlyStationPressure',
                            'origin_3Hr_StationPressure','dest_3Hr_StationPressure',
                            'origin_6Hr_StationPressure','dest_6Hr_StationPressure',
                            'origin_12Hr_StationPressure','dest_12Hr_StationPressure',
                            'origin_DailyAverageStationPressure','dest_DailyAverageStationPressure']

otpw_1y = otpw_1y.dropna(subset=weather_feats_drop_nulls)
funcs.get_df_dimensions(otpw_1y)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop feature entirely due to nulls: dest_HourlyWindDirection, origin_HourlyWindDirection 

# COMMAND ----------


drop_entire_weather_feats = ['origin_HourlyWindDirection','dest_HourlyWindDirection']
funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*drop_entire_weather_feats)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# # add 'day_of_year' column
# otpw_1y = otpw_1y.withColumn('day_of_year', F.dayofyear(F.col('FL_DATE')))


# now we can drop the unused data time columns 
unused_date_cols = ['FL_DATE', #information already contained in 'sched_depart_date_time_UTC'
                    'QUARTER', # less granular than month & can easily be extracted from date if needed
                    'DAY_OF_MONTH', # less useful than day of year & day of week + already contained in date
                    'YEAR'] # same as above 

funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.drop(*unused_date_cols)
funcs.get_df_dimensions(otpw_1y)


# COMMAND ----------

# MAGIC %md
# MAGIC #### visibility features with missing values: drop null observations 

# COMMAND ----------

funcs.get_df_dimensions(otpw_1y)

visibility_feats_drop_nulls = ['origin_HourlyVisibility','dest_HourlyVisibility',
                            'origin_3Hr_Visibility','dest_3Hr_Visibility',
                            'origin_6Hr_Visibility','dest_6Hr_Visibility',
                            'origin_12Hr_Visibility','dest_12Hr_Visibility']


otpw_1y = otpw_1y.dropna(subset=visibility_feats_drop_nulls)
funcs.get_df_dimensions(otpw_1y)


# COMMAND ----------

funcs.get_df_dimensions(otpw_1y)

wind_feats_drop_nulls = ['dest_DailySustainedWindDirection','origin_DailySustainedWindDirection',
                         'dest_DailySustainedWindSpeed','origin_DailySustainedWindSpeed']

otpw_1y = otpw_1y.dropna(subset=wind_feats_drop_nulls)
funcs.get_df_dimensions(otpw_1y)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Flight features: remaining nulls

# COMMAND ----------

end_of_eda3_missingdf = funcs.count_missing_values(otpw_1y)
funcs.filter_df_by_min_max(end_of_eda3_missingdf,'null_percent',0.00001)

# COMMAND ----------

# for now, drop reamining nulls (diverted obs)
funcs.get_df_dimensions(otpw_1y)
otpw_1y = otpw_1y.dropna()
funcs.get_df_dimensions(otpw_1y)


# COMMAND ----------

# write data as parquet file to storage 
otpw_1y.write.mode('overwrite').parquet(f"{team_blob_url}/LH/1yr_clean_temp_2")

# COMMAND ----------


