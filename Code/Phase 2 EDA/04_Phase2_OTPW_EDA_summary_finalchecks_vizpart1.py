# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 EDA: Re-joined 1-year OTPW Dataset
# MAGIC
# MAGIC **Team 5-3: Bailey Kuehl, Lucy Herr, Artem Lebedev, Erik Sambraillo** 
# MAGIC <br>**261 Fall 2023**
# MAGIC <br>
# MAGIC <br>
# MAGIC This notebook (04) summarizes the EDA process for the 1-year OTPW dataset (2015) and visualizes some of the key variable relationships in the data. This version of the data has been re-joined from the raw flights and weather tables in order to more accurately and thoroughly retain the original weather feature values.
# MAGIC
# MAGIC ### EDA components for Phase 2 report
# MAGIC - A data dictionary of the raw features (test description; data type: numerical, list, etc.)
# MAGIC - Dataset size (rows columns, train, test, validation)
# MAGIC - Summary statistics
# MAGIC - Correlation analysis
# MAGIC - Other useful text-based analysis (as opposed to graphic-based)
# MAGIC - A visualization of **each of the input and target features** (looking at the distribution, and the central tendencies as captured by the mean, median etc.)
# MAGIC - A visualization of the correlation analysis
# MAGIC - pair-based visualization of the input and output features
# MAGIC - a graphic summary of the missing value analysis
# MAGIC -- other novel visualizations (e.g., geo-plot)
# MAGIC
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

from datetime import datetime

# COMMAND ----------

# connect to team blob
team_blob_url = funcs.blob_connect()
# view blob storage root folder 
display(dbutils.fs.ls(f"{team_blob_url}/LH"))

# COMMAND ----------

# current version otpw 1yr
otpw_1y = spark.read.parquet(f"{team_blob_url}/LH/1yr_clean_temp_2")
funcs.get_df_dimensions(otpw_1y)
otpw_1y.columns

# COMMAND ----------



# COMMAND ----------

otpw_1y.dtypes

# COMMAND ----------

# airline name reference table (for readable airport names in plots)
airline_names_df = spark.read.parquet(f"{team_blob_url}/airline_names_df")
display(airline_names_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary of EDA & data cleaning process performed up to this point: 
# MAGIC 1. Address missing values: review features with high proportions of missing values and drop
# MAGIC 2. Feature selection: evaluate and remove any features that don't contribute additional information to prediction in order to increase efficiency of working with the data in EDA and modeling. Whenever features (e.g., airport names) are not relevant to modeling but potentially useful for visualization and reference, create and store look-up tables for future reference. 
# MAGIC 3. Imputation: when possible, impute remaining missing values based on domain input 
# MAGIC 4. Remove any observations that can't contribute to prediction 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Re-joined 1-Year OTPW Data Cleaning 
# MAGIC
# MAGIC **Dimensions of raw re-joined OTPW 1-year data: (5812241,162)**
# MAGIC
# MAGIC ### Data cleaning summary table 
# MAGIC | Subset description | Example feature(s) | Subset Size| Decision | Resulting dimensions | 
# MAGIC | ---------- | ------------| ---------- | ---------- | ---------- |
# MAGIC | features with >90% null values  |`LONGEST_ADD_GTIME`,`FIRST_DEP_TIME`,`CANCELLATION_CODE` | (,4)| drop | (5812241,158) |
# MAGIC | specific "delay type" features with 80+% nulls  | `CARRIER_DELAY`,`LATE_AIRCRAFT_DELAY` | (,5) | drop | (5812241,153)|
# MAGIC | WindGustSpeed features with 66-88% null values | `dest_HourlyWindGustSpeed`,`origin_12Hr_WindGustSpeed` | (,8) | drop | (5812241,145) |
# MAGIC | HourlyPressureChange features with ~67% null values | `dest_HourlyPressureChange`, `origin_HourlyPressureChange` | (,2)| drop | (5812241,143)|
# MAGIC | cancelled flight observations (~1.5% total) | `CANCELLED`==1 | (89788,)| drop | (5722453,143) |
# MAGIC | DailySnowfall features with 25-26% null values | `dest_DailySnowfall`,`origin_DailySnowfall` |(1513685,2)| impute 0 (no snowfall) | unchanged | 
# MAGIC | redundant/descriptive weather station features |`origin_station_lat`,`'dest_station_name`| (,10)|write to lookup table & drop | (5722453,133)|
# MAGIC | redundant/descriptive airport features | `ORIGIN_CITY_NAME`,`DEST_STATE_FIPS`,`DEST_STATE_NM` | (,10)| write to lookup table & drop | (5722453,123)|
# MAGIC | less granular departure delay features | `DEP_DELAY_NEW`,`DEP_DELAY_GROUP`,`DEP_TIME_BLK` |(,3)| drop - retain `DEP_DELAY` | (5722453, 120) | 
# MAGIC | less granular arrival delay features | `ARR_DELAY_NEW`,`ARR_DELAY_GROUP`,`ARR_TIME_BLK` |(,3)| drop - retain `ARR_DELAY` | (5722453, 117) | 
# MAGIC | flights features with uniform values | `FLIGHTS`,`CANCELLED`| (,2)| drop - all values the same | (5722453,115)|
# MAGIC | redundant flight time columns | `WHEELS_OFF`,`WHEELS_ON` | (,2)| drop  - redundant to `TAXI_OUT`/`TAXI_IN` + `DEP_TIME`/`ARR_TIME` | (5722453,113) |
# MAGIC | extraneous date/time columns | `FL_DATE`,`QUARTER`,`DAY_OF_MONTH`,`YEAR` | (,4) | drop | (5722453,109) |
# MAGIC | engineered date/time columns  | `day_of_year` | (,1) | create + add |  (5722453,110) |
# MAGIC | remaining weather columns with nulls, subset 1 | `origin_HourlyPrecipitation`, `origin_3Hr_PressureChange` | (,20) | impute as 0 | unchanged |
# MAGIC | remaining weather columns with nulls, subset 2 | `dest_HourlyDryBulbTemperature`,`origin_HourlyVisibility` | (40) | drop null observations | (4449286,110) |
# MAGIC | remaining weather columns with nulls, subset 3 |`origin_HourlyWindDirection`,`dest_HourlyWindDirection` | (,2) | drop features| (4449286,108)|
# MAGIC | "diverted" observations | `DIVERTED`==1 | (11479,) | drop small proportion of obs. (.25%) w/ nulls in flight features|(4437807,107)|
# MAGIC
# MAGIC
# MAGIC **Dimensions of cleaned re-joined OTPW 1-year data: 4437807,107**
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary table of missing data
# MAGIC
# MAGIC | Feature | Group |  Type | Description | Percent Missing | Decision |
# MAGIC | -------- | ------ | ------- | -------- | -------- | -------- |
# MAGIC | **`ARR_DEL15`** |Flight Data| flight delayed by at least 15 minutes: 0=no, 1=yes | indicator |   0.27  | primarily diverted flights - drop these obs. at end of EDA NB03  | 
# MAGIC | **`ARR_DELAY`** |Flight Data| length of flight delay in minutes (early flights = negative values) | numeric (minutes) |  0.27  | primarily diverted flights - drop these obs. at end of EDA NB03 | 
# MAGIC | **`TAXI_IN`** |Flight Data| taxi-ing time between gate and takeoff | numeric (minutes) |  0.05  | primarily diverted flights - drop these obs. at end of EDA NB03 |
# MAGIC | **`ARR_TIME`** |Flight Data| actual flight arrival time | local time HHMM |  0.05  | primarily diverted flights - drop these obs. at end of EDA NB03 |
# MAGIC | **`ACTUAL_ELAPSED_TIME`** |Flight Data| actual flight duration |numeric (minutes) |   0.27  | primarily diverted flights - drop these obs. at end of EDA NB03 |
# MAGIC | **`CRS_ELAPSED_TIME`** |Flight Data| scheduled flight duration  | numeric (minutes) |   0.00002 | primarily diverted flights - drop these obs. at end of EDA NB03 |
# MAGIC | **`AIR_TIME`**|Flight Data| Flight Time, in Minutes| numeric (minutes) |0.27  | primarily diverted flights - drop these obs. at end of EDA NB03 |
# MAGIC | `origin_HourlyPrecipitation` |Precipitation|| | 1.73  | Impute with 0's |
# MAGIC | `dest_HourlyPrecipitation` |Precipitation|| |  1.84  | Impute with 0's |
# MAGIC | `origin_3Hr_Precipitation` |Precipitation|| |  1.45  | Impute with 0's |
# MAGIC | `dest_3Hr_Precipitation` |Precipitation|| |  1.57   | Impute with 0's |
# MAGIC | `origin_6Hr_Precipitation` |Precipitation|| |  1.34  | Impute with 0's|
# MAGIC | `dest_6Hr_Precipitation` |Precipitation|| |  	1.47  | Impute with 0's |
# MAGIC | `origin_12Hr_Precipitation` |Precipitation|| |   1.19   | Impute with 0's|
# MAGIC | `dest_12Hr_Precipitation` |Precipitation|| | 1.32    | Impute with 0's |
# MAGIC | `origin_DailyPrecipitation` |Precipitation|| |  1.40  | Impute with 0's |
# MAGIC | `dest_DailyPrecipitation` |Precipitation|| |  1.52  | Impute with 0's |
# MAGIC | `origin_DailyDepartureFromNormalAverageTemperature` |Temperature| | |  3.51  | Impute with 0's |
# MAGIC | `dest_DailyDepartureFromNormalAverageTemperature ` |Temperature|| |  3.63  | Impute with 0's |
# MAGIC | `origin_HourlyDryBulbTemperature` |Temperature| | |  0.21  | Drop missing values|
# MAGIC | `dest_HourlyDryBulbTemperature` |Temperature| | |	0.31| Drop missing values| 
# MAGIC | `origin_3Hr_DryBulbTemperature` |Temperature| | |  0.17  | Drop missing values|
# MAGIC | `dest_3Hr_DryBulbTemperature`|Temperature| | | 	0.27  | Drop missing values| 
# MAGIC | `origin_6Hr_DryBulbTemperature` |Temperature| | |   0.17  | Drop missing values|
# MAGIC | `dest_6Hr_DryBulbTemperature`|Temperature| | |	0.27  | Drop missing values|
# MAGIC | `origin_12Hr_DryBulbTemperature` |Temperature| | | 0.16  |Drop missing values |
# MAGIC | `dest_12Hr_DryBulbTemperature` |Temperature| | |  0.26  | Drop missing values|
# MAGIC | `origin_DailyAverageDryBulbTemperature` |Temperature|| |   1.40 | Drop missing values |
# MAGIC | `dest_DailyAverageDryBulbTemperature` |Temperature|| |  1.52  | Drop missing values |
# MAGIC | `origin_DailyAverageRelativeHumidity` |Humidity| | |  14.83   | Drop missing values	|
# MAGIC | `dest_DailyAverageRelativeHumidity` |Humidity|| |  14.95   | Drop missing values	|
# MAGIC | `origin_HourlyStationPressure`| Pressure| | | 0.39| Drop missing values |
# MAGIC | `dest_HourlyStationPressure`|Pressure| | | 0.51| Drop missing values |
# MAGIC | `origin_3Hr_StationPressure`|Pressure| | |	0.27  | Drop missing values| 
# MAGIC | `dest_3Hr_StationPressure` |Pressure| | |0.38 | Drop missing values|
# MAGIC | `origin_6Hr_StationPressure` |Pressure| | |  0.23  | Drop missing values|
# MAGIC | `dest_6Hr_StationPressure` |Pressure| | |	0.35 | Drop missing values| 
# MAGIC | `origin_12Hr_StationPressure` | Pressure|| | 0.21  | Drop missing values| 
# MAGIC | `dest_12Hr_StationPressure` |Pressure| | |	0.33 | Drop missing values|
# MAGIC | `origin_DailyAverageStationPressure` |Pressure|| |  	1.40  | Drop missing values |
# MAGIC | `dest_DailyAverageStationPressure` |Pressure|| | 1.52 | Drop missing values |
# MAGIC | `origin_3Hr_PressureChange` |Pressure|| |  1.30  | Impute with 0's|
# MAGIC | `dest_3Hr_PressureChange` |Pressure|| |  1.38  |Impute with 0's|
# MAGIC | `origin_6Hr_PressureChange`|Pressure|| | 0.85| Impute with 0's|
# MAGIC | `dest_6Hr_PressureChange` |Pressure|| | 0.96|	Impute with 0's|
# MAGIC | `origin_12Hr_PressureChange` |Pressure| | | 0.71| Impute with 0's| 
# MAGIC | `dest_12Hr_PressureChange` |Pressure|| | 0.83 | Impute with 0's|
# MAGIC | `origin_HourlyWindDirection` |Wind| | |  4.52  | Drop entirely. Not relevant without windspeed |
# MAGIC | `dest_HourlyWindDirection` |Wind|| |  4.67  | Drop entirely. Not relevant without windspeed	|
# MAGIC | `origin_DailySustainedWindSpeed` |Wind|| |  1.32  |  Drop missing values|
# MAGIC | `origin_DailySustainedWindDirection` |Wind|| |  1.32  | Drop missing values|
# MAGIC | `dest_DailySustainedWindSpeed` |Wind|| |  1.44  | Drop missing values |
# MAGIC | `dest_DailySustainedWindDirection` |Wind|| |  1.44  | Drop missing values |
# MAGIC | `origin_HourlyVisibility` | Visibility| | |	0.32| Drop missing values | 
# MAGIC | `dest_HourlyVisibility`| Visibility| | | 0.41| Drop missing values |
# MAGIC | `origin_3Hr_Visibility` |Visibility| | |  0.21  | Drop missing values | 
# MAGIC | `dest_3Hr_Visibility`|Visibility| | | 0.31 | Drop missing values | 
# MAGIC | `origin_6Hr_Visibility` |Visibility| | |  0.17  | Drop missing values |
# MAGIC | `dest_6Hr_Visibility`|Visibility| | | 	0.27  | Drop missing values | 
# MAGIC | `origin_12Hr_Visibility` |Visibility| | |   0.16  | Drop missing values |
# MAGIC | `dest_12Hr_Visibility` |Visibility| | |   0.26  | Drop missing values |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Remaining feature checks:
# MAGIC
# MAGIC 1. **Check OP UNIQUE CARRIER for inconsistencies over time using other carrier vars to see if we can use this feature alone to identify carriers**
# MAGIC   - EDA finding: there are no inconsistencies (more than one distinct associated value in the other 2 features) for 'OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID',	and 'OP_CARRIER'
# MAGIC   - Decision: drop 'OP_CARRIER_AIRLINE_ID'	and 'OP_CARRIER' in 2015 data, but plan on re-checking the 4-year data as this may not be the case over multiple years
# MAGIC
# MAGIC 2. **Does City market ID ('ORIGIN_CITY_MARKET_ID'/'DEST_CITY_MARKET_ID') contribute useful information (e.g., is the number of airports per metro area related to departure delay such that it contributes information gain beyond origin airport characteristics alone)?**
# MAGIC   - EDA finding: average dep delay for city market ID is almost perfectly correlated with average dep delay for associated origin airport. 
# MAGIC   - Decision: drop city market ID features (2) because they don't contribute much information beyond the constituent origin airports 
# MAGIC
# MAGIC 3. **Does origin U.S. state contribute useful information about delays?**
# MAGIC   - EDA Finding: Pearson's correlation coefficient for average departure delay by origin and and average departure delay by origin state (0.62) falls short of the 0.70 threshold that indicates multicollinearity, so there may be some useful info gain. 
# MAGIC   - Decision: Keep `ORIGIN_STATE_ABR` and `DEST_STATE_ABR` in the modeling data for now and evaluate them in more depth with our decision tree model.
# MAGIC
# MAGIC 4. **ICAO: does this contribute useful info about delays?**
# MAGIC   - EDA finding: each airport has only 1 unique ICAO associated with it
# MAGIC   - Decision: drop 'origin_icao'/'dest_icao' - redundant to airport 
# MAGIC
# MAGIC 5. **Confirm that we can use Origin/Destination airport ID interchangably with 'ORIGIN'/'DEST'**
# MAGIC   - Airport ID definition: "An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.
# MAGIC   - compare 'ORIGIN' vs. 'ORIGIN_AIRPORT_ID' and  'DEST' vs. 'DEST_AIRPORT_ID'
# MAGIC   - EDA findings: in the 2015 dataset, both origin airport ID and destination airport ID have one-to-one relationships with origin/destination airport, so can continue to use `ORIGIN` and `DEST` as our unique airport identifiers (at least in the 2015 data). 
# MAGIC   - Decision: OK to drop the ID cols from the 1year data; **however, we should definitely confirm that this relationship holds up in 4-year data before discarding the airport ID features.**
# MAGIC
# MAGIC 6. **International vs. non-international airports vs. airport "type" (Small/Med/Large") vs. delays: do these add info?**
# MAGIC   - EDA Findings: After parsing "origin_airport_name" and "dest_airport_name" to compare average delay times by international vs. non-international origins airports, we observed only negligible differences. Howevver, average delay by origin & average delay by origin_type aren't as highly correlated as we would expect (0.33).
# MAGIC   - Decisions: drop 'origin_airport_name' & 'dest_airport_name', retain 'origin_type' & 'dest_type' for now
# MAGIC
# MAGIC 7. **Confirm that `origin_station_id` & `dest_station_id` are redundant to airport (one-to-one airport-station relationship):**
# MAGIC - EDA finding:  `origin_station_id`/`dest_station_id` and `origin_station_dis`/`dest_station_dis`have no info gain against knowing the airport alone.
# MAGIC - Decision: drop all 4
# MAGIC
# MAGIC 8. **drop 'DIVERTED' (all values=0 since we dropped diverted flight observations) & 'DISTANCE_GROUP' (less granular than distance)**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Check OP UNIQUE CARRIER for inconsistencies over time using other carrier vars to see if we can use this feature alone to identify carriers
# MAGIC   - EDA finding: there are no inconsistences (more than one distinct associated value in the other 2 features) for 'OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID',	and 'OP_CARRIER'
# MAGIC   - Decision: drop 'OP_CARRIER_AIRLINE_ID'	and 'OP_CARRIER' in 2015 data, but plan on re-checking the 4-year data as this may not be the case over multiple years

# COMMAND ----------

funcs.funcs.print_all_distinct_values(otpw_1y, 'OP_UNIQUE_CARRIER')

# COMMAND ----------

funcs.print_all_distinct_values(otpw_1y, 'OP_CARRIER_AIRLINE_ID')

# COMMAND ----------

funcs.print_all_distinct_values(otpw_1y, 'OP_CARRIER')

# COMMAND ----------


#moved to funcs file
# def check_column_consistency(df: DataFrame, col1: str, col2: str, col3: str = None) -> DataFrame:
#     """
#     Checks if the values in three specified columns of a Spark DataFrame are consistent.

#     Parameters:
#     df (DataFrame): The Spark DataFrame to check.
#     col1 (str): The name of the first column to compare.
#     col2 (str): The name of the second column to compare.
#     col3 (str): (OPTIONAL) The name of the third column to compare.

#     Returns:
#     DataFrame: A DataFrame containing the inconsistencies, if any.
#     """
#     # find unique combinations of the specified columns
#     unique_combinations = df.select(col1, col2, col3).distinct()
#     # count each combination
#     combination_counts = unique_combinations.groupBy(col1, col2, col3).count()
#     # filter for combinations where count > 1
#     inconsistencies = combination_counts.filter(F.col("count") > 1)
#     return inconsistencies

carrier_check_df = funcs.check_column_consistency(otpw_1y, "OP_UNIQUE_CARRIER", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER")
carrier_check_df.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Looks good, so we can drop OP_CARRIER_AIRLINE_ID and OP_CARRIER - but we should re-check this in the 4-year data. 

# COMMAND ----------

#  drop OP_CARRIER_AIRLINE_ID and OP_CARRIER
funcs.get_df_dimensions(otpw_1y)
drop_carrier_cols = ['OP_CARRIER_AIRLINE_ID','OP_CARRIER']
otpw_1y = otpw_1y.drop(*drop_carrier_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Does City market ID ('ORIGIN_CITY_MARKET_ID'/'DEST_CITY_MARKET_ID') contribute useful information (e.g., is the number of airports per metro area related to departure delay such that it contributes information gain beyond origin airport dealy characteristics alone)?
# MAGIC - EDA finding: average dep delay for city market ID is almost perfectly correlated with average dep delay for associated origin airport.
# MAGIC - Decision: drop city market ID features (2) because they don't contribute much information beyond the constituent origin airports
# MAGIC

# COMMAND ----------

# total distinct city markets: 
distinct_org_city_markets = otpw_1y.select("ORIGIN_CITY_MARKET_ID").distinct().count()
print(f"Number of distinct Origin City Markets': {distinct_org_city_markets}")

# COMMAND ----------

# plot avg departure delay per city market 
# groupby 'ORIGIN_CITY_MARKET_ID' and calculate average 'DEP_DELAY'
avg_delay_df = otpw_1y.groupBy("ORIGIN_CITY_MARKET_ID").agg(F.avg("DEP_DELAY").alias("AVG_DEP_DELAY"))
# convert to pd dffor plotting
pandas_df = avg_delay_df.toPandas()
# sort results by "AVG_DEP_DELAY"
pandas_df = pandas_df.sort_values(by="AVG_DEP_DELAY")

# plot
plt.figure(figsize=(10, 6))
plt.bar(pandas_df["ORIGIN_CITY_MARKET_ID"], pandas_df["AVG_DEP_DELAY"])
plt.xlabel("Origin City Market ID")
plt.ylabel("Average Departure Delay (minutes)")
plt.title("Average Departure Delay per City Market")
plt.gca().set_xticklabels([])
plt.tight_layout()  # Adjust layout for better fit
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Average departure delay by origin city market shows a substantial degree of varition (ranging from approx. -6 minutes to ~25 minutes of delay). Does this range primarily reflect the delays of the associated origin airports, or the city market attributes themselves? 

# COMMAND ----------

# number of origin airports per city market ID
# Group by 'ORIGIN_CITY_MARKET_ID', collect unique 'ORIGIN' values into a list, and count them
origin_airports_per_market_id = otpw_1y.groupBy("ORIGIN_CITY_MARKET_ID")\
    .agg(
        F.collect_set("ORIGIN").alias("ORIGIN_AIRPORTS"),
        F.size(F.collect_set("ORIGIN")).alias("AIRPORT_COUNT")
    )\
    .orderBy("AIRPORT_COUNT",ascending=False)

# Show the result
origin_airports_per_market_id.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Only a handful of city markets have more than one origin airport assoicated with them: New York City (4), Washington DC (3), Dallas (2), Cleveland/Akron (2), Houston (2), SF Bay/Peninsula (2), Miami/Ft. Lauderdale (2), and LA (2). Because the vast majority of the 220 total city markets have only one airport, I'm skeptical that this feature will offer much information gain, but let's confirm this. 

# COMMAND ----------

#origin_citymrkt_comp_delay_df = otpw_1y.select("ORIGIN","ORIGIN_CITY_MARKET_ID").distinct()

# calculate average delay by origin airport
avg_delay_by_origin = otpw_1y.groupBy("ORIGIN").agg(F.avg("DEP_DELAY").alias("AVG_DELAY_ORIGIN"))

# calculate average departure delay by city market ID
avg_delay_by_city_market = otpw_1y.groupBy("ORIGIN_CITY_MARKET_ID").agg(F.avg("DEP_DELAY").alias("AVG_DELAY_CITY_MARKET"))

combined_df = origin_citymrkt_comp_delay_df.join(avg_delay_by_origin, "ORIGIN")
combined_df = combined_df.join(avg_delay_by_city_market, "ORIGIN_CITY_MARKET_ID")

correlation = combined_df.stat.corr("AVG_DELAY_ORIGIN", "AVG_DELAY_CITY_MARKET")

print("Pearson's correlation coefficient between AVG_DELAY_ORIGIN AVG_DELAY_CITY_MARKET: ", correlation)

# COMMAND ----------

# MAGIC %md
# MAGIC As city market ID is almost perfectly correlated with origin in terms of average departure delay, we can drop it without being too concerned about information loss. 

# COMMAND ----------

#  drop city market ID columns
funcs.get_df_dimensions(otpw_1y)
drop_citymarket_cols = ['ORIGIN_CITY_MARKET_ID','DEST_CITY_MARKET_ID']
otpw_1y = otpw_1y.drop(*drop_citymarket_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Does origin U.S. state contribute useful information about delays? 
# MAGIC   - EDA Finding: Pearson's correlation coefficient for average departure delay by origin and and average departure delay by origin state (0.62) falls short of the 0.70 threshold that indicates multicollinearity, so there may be some useful info gain. 
# MAGIC   - Decision: Keep `ORIGIN_STATE_ABR` and `DEST_STATE_ABR` in the modeling data for now and evaluate them in more depth with our decision tree model.

# COMMAND ----------

# plot avg departure delay by U.S. state
# groupby 'state' and calculate average 'DEP_DELAY'
avg_state_delay_df = otpw_1y.groupBy("ORIGIN_STATE_ABR").agg(F.avg("DEP_DELAY").alias("AVG_DEP_DELAY_BY STATE"))
# convert to pd dffor plotting
avg_state_delay_df = avg_state_delay_df.toPandas()
# sort results by "AVG_DEP_DELAY"
avg_state_delay_df = avg_state_delay_df.sort_values(by="AVG_DEP_DELAY_BY STATE")

# plot results
plt.figure(figsize=(10, 6))
plt.bar(avg_state_delay_df["ORIGIN_STATE_ABR"], avg_state_delay_df["AVG_DEP_DELAY_BY STATE"])
plt.xlabel("Origin StatD")
plt.ylabel("Average Departure Delay (minutes)")
plt.title("Average Departure Delay per State")
plt.gca().set_xticklabels([])
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Wow. What is that outlier state? 

# COMMAND ----------

avg_state_delay_df = avg_state_delay_df.sort_values(by="AVG_DEP_DELAY_BY STATE")
avg_state_delay_df.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC So Delaware had extremely high average delay times (~26 minutes) in 2015, while all other states fell within ~2-12 minutes. Again, the vast majority of the states seem to fall within a fairly consistent range, so it's doubtful that this feature will offer much information gain, but let's confirm this.  

# COMMAND ----------

origin_state_comp_df = otpw_1y.select("ORIGIN","ORIGIN_STATE_ABR").distinct()

# calculate average delay by state
avg_delay_by_origin = otpw_1y.groupBy("ORIGIN").agg(F.avg("DEP_DELAY").alias("AVG_DELAY_ORIGIN"))

# calculate average departure delay by city market ID
avg_delay_by_state = otpw_1y.groupBy("ORIGIN_STATE_ABR").agg(F.avg("DEP_DELAY").alias("AVG_DELAY_ORIGIN_STATE"))

combined_df = origin_state_comp_df.join(avg_delay_by_origin, "ORIGIN")
combined_df = combined_df.join(avg_delay_by_state, "ORIGIN_STATE_ABR")

correlation = combined_df.stat.corr("AVG_DELAY_ORIGIN", "AVG_DELAY_ORIGIN_STATE")

print("Pearson's correlation coefficient between AVG_DELAY_ORIGIN & AVG_DELAY_ORIGIN_STATE: ", correlation)

# COMMAND ----------

# MAGIC %md
# MAGIC Pearson's correlation coefficient for average departure delay by origin and and average departure delay by origin state (0.62) falls short of the 0.70 threshold that indicates multicollinearity, so there may be some useful info gain from retaining these features and evaluating their predictive utility in more depth with our decision tree model. Let's tentatively leave them in for now.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. ICAO: do 'origin_icao'/'dest_icao' contribute unique info about delays?
# MAGIC   - EDA finding: each airport has only 1 unique ICAO associated with it
# MAGIC   - Decision: drop 'origin_icao'/'dest_icao' - redundant to airport 

# COMMAND ----------

# number of origin airports per ICAO
# Group by 'ORIGIN_ICAO', collect unique 'ORIGIN' values into a list, and count them
origin_airports_per_icao = otpw_1y.groupBy("origin_icao")\
    .agg(
        F.collect_set("ORIGIN").alias("ORIGIN_AIRPORTS"),
        F.size(F.collect_set("ORIGIN")).alias("AIRPORT_COUNT")
    )\
    .orderBy("AIRPORT_COUNT",ascending=False)

# Show the result
origin_airports_per_icao.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Given that each airport has only one associated ICAO value, we can safely drop the ICAO features given that they are otherwise represented in the distinct airport values. The plot of average delay by ICAO should also align with the plot of average delay by state:

# COMMAND ----------

# plot avg departure delay by ICAO
origin_ICAO_comp_df = otpw_1y.select("ORIGIN","origin_icao").distinct()

# groupby 'icao' and calculate average 'DEP_DELAY'
avg_ICAO_delay_df = otpw_1y.groupBy("origin_icao").agg(F.avg("DEP_DELAY").alias("AVG_DEP_DELAY_BY_origin_icao"))
# convert to pd df for plotting
avg_ICAO_delay_df = avg_ICAO_delay_df.toPandas()
# sort results by avg. ICAO delay
avg_ICAO_delay_df = avg_ICAO_delay_df.sort_values(by="AVG_DEP_DELAY_BY_origin_icao")

# plot results
plt.figure(figsize=(10, 6))
plt.bar(avg_ICAO_delay_df["origin_icao"], avg_ICAO_delay_df["AVG_DEP_DELAY_BY_origin_icao"])
plt.xlabel("Origin ICAO")
plt.ylabel("Average Departure Delay (minutes)")
plt.title("Average Departure Delay per ICAO")
plt.gca().set_xticklabels([])
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Looks right. Let's drop the ICAO features ('origin_icao' & 'dest_icao')

# COMMAND ----------

#  drop OP_CARRIER_AIRLINE_ID and OP_CARRIER
funcs.get_df_dimensions(otpw_1y)
drop_icao_cols = ['origin_icao','dest_icao']
otpw_1y = otpw_1y.drop(*drop_icao_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Confirm that we can use Origin/Destination airport ID interchangably with 'ORIGIN'/'DEST'
# MAGIC   - Airport ID definition: "An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.
# MAGIC   - compare 'ORIGIN' vs. 'ORIGIN_AIRPORT_ID'	
# MAGIC   - compare 'DEST' vs. 'DEST_AIRPORT_ID'

# COMMAND ----------

# compare 'ORIGIN' vs. 'ORIGIN_AIRPORT_ID'
# number of origin airport ids per 'ORIGIN'
# Group by 'ORIGIN', collect unique 'ORIGIN_AIRPORT_D' values into a list, and count them
origins_per_origin_airport_id = otpw_1y.groupBy("ORIGIN_AIRPORT_ID")\
    .agg(
        F.collect_set("ORIGIN").alias("ORIGIN_AIRPORTS"),
        F.size(F.collect_set("ORIGIN")).alias("AIRPORT_COUNT")
    .orderBy("AIRPORT_COUNT",ascending=False)

# Show the result
origins_per_origin_airport_id.show(truncate=False)	

# COMMAND ----------

# reverse the comparison: compare 'ORIGIN_AIRPORT_ID' vs. 'ORIGIN'
# number of origin airport ids per 'ORIGIN'
# Group by 'ORIGIN', collect unique 'ORIGIN_AIRPORT_ID' values into a list, and count them
origin_airport_ids_per_origin = otpw_1y.groupBy("ORIGIN")\
    .agg(
        F.collect_set("ORIGIN_AIRPORT_ID").alias("ORIGIN_AIRPORT_IDs"),
        F.size(F.collect_set("ORIGIN_AIRPORT_ID")).alias("origin_airport_id_count")
    .orderBy("origin_airport_id_count",ascending=False)
    )

# show the result
origin_airport_ids_per_origin.show(truncate=False)

# COMMAND ----------

# repeat comparision for 'DEST' vs. 'DEST_AIRPORT_ID'(being thorough here because these need to serve as unique identifiers for any FE)

# Group by 'DEST', collect unique 'DEST_AIRPORT_D' values into a list, and count them
dests_per_dest_airport_id = otpw_1y.groupBy("DEST_AIRPORT_ID")\
    .agg(
        F.collect_set("DEST").alias("DEST_AIRPORTS"),\
        F.size(F.collect_set("DEST")).alias("AIRPORT_COUNT")\
            )\
    .orderBy("AIRPORT_COUNT",ascending=False)

dests_per_dest_airport_id.show(truncate=False)	


# COMMAND ----------

# MAGIC %md
# MAGIC Given that both origin airport ID and destination airport is have one-to-one relationships with origin/destination airport, we can continue to use `ORIGIN` and `DEST` as our unique airport identifiers (at least in the 2015 data). However, we should definitely confirm that this relationship holds up in 4-year data before discarding the airport ID features.

# COMMAND ----------

# drop AIRPORT ID cols *2015 DATA ONLY&
funcs.get_df_dimensions(otpw_1y)
drop_airport_id_cols = ['ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID']
otpw_1y = otpw_1y.drop(*drop_airport_id_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC 6. **Do international airports have different average delays than non-international airports? What about airport "type" (Small/Med/Large")?**
# MAGIC   - EDA Finding: After parsing "origin_airport_name" and "dest_airport_name" to compare average delay times by international vs. non-international origins airports, we observed only negligible differences. Howevver, average delay by origin & average delay by origin_type aren't as highly correlated as we would expect (0.33)/ 
# MAGIC   - Decision: drop 'origin_airport_name'	& 'dest_airport_name', retain 'origin_type' & 'dest_type' for now

# COMMAND ----------

# Create a new column 'is_international'
origin_intl_delay = otpw_1y.select("origin_airport_name", "DEP_DELAY","DEP_DEL15").withColumn('is_international', F.when(F.col('origin_airport_name').contains('International'), 1).otherwise(0))

origin_intl_delay = origin_intl_delay.groupBy('is_international').agg(F.avg("DEP_DELAY").alias("AVG_DELAY"))
# Show the result
origin_intl_delay.show()

# COMMAND ----------

# Create a new column 'is_international'
dest_intl_delay = otpw_1y.select("dest_airport_name", "ARR_DELAY","ARR_DEL15").withColumn('is_international', F.when(F.col('dest_airport_name').contains('International'), 1).otherwise(0))

dest_intl_delay = dest_intl_delay.groupBy('is_international').agg(F.avg("ARR_DELAY").alias("AVG_DELAY"))
# Show the result
dest_intl_delay.show()

# COMMAND ----------

# MAGIC %md
# MAGIC As the difference in average delay times between international airports and non-international origin airports is negligible (as well as average arrival delay for destination airports), let's drop these. 

# COMMAND ----------

# drop AIRPORT name cols
funcs.get_df_dimensions(otpw_1y)
drop_airport_name_cols = ["origin_airport_name","dest_airport_name"]
otpw_1y = otpw_1y.drop(*drop_airport_name_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's compare average delay times by `origin_type` & `dest_type`:

# COMMAND ----------

# check values in origin type
funcs.print_all_distinct_values(otpw_1y, 'origin_type')

# COMMAND ----------

# check values in dest type
funcs.print_all_distinct_values(otpw_1y, 'dest_type')

# COMMAND ----------

origin_type_avg_delay = otpw_1y.groupBy('origin_type').agg(F.avg("DEP_DELAY").alias("AVG_DELAY"))
origin_type_avg_delay.show()

# COMMAND ----------

dest_type_avg_delay = otpw_1y.groupBy('dest_type').agg(F.avg("DEP_DELAY").alias("AVG_DELAY"))
dest_type_avg_delay.show()

# COMMAND ----------

# check correlation with origin airports

origin_type_comp = otpw_1y.select("ORIGIN","origin_type").distinct()

# calculate average delay by origin airport
avg_delay_by_origin = otpw_1y.groupBy("ORIGIN").agg(F.avg("DEP_DELAY").alias("AVG_DELAY_ORIGIN"))

# calculate average departure delay by city market ID
avg_delay_by_type = otpw_1y.groupBy("origin_type").agg(F.avg("DEP_DELAY").alias("AVG_DELAY_ORIGIN_TYPE"))

combined_df = origin_type_comp .join(avg_delay_by_origin, "ORIGIN")
combined_df = combined_df.join(avg_delay_by_type, "ORIGIN_TYPE")

correlation = combined_df.stat.corr("AVG_DELAY_ORIGIN", "AVG_DELAY_ORIGIN_TYPE")

print("Pearson's correlation coefficient between AVG_DELAY_ORIGIN & AVG_DELAY_ORIGIN_TYPE: ", correlation)

# COMMAND ----------

# MAGIC %md
# MAGIC  AVG_DELAY_ORIGIN & AVG_DELAY_ORIGIN_TYPE aren't as highly correlated as we would expect - let's retain these for now (to evaluate with DT model). 

# COMMAND ----------

funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7. Confirm that `origin_station_id` & `dest_station_id` are redundant to airport (one-to-one airport-station relationship):
# MAGIC - EDA finding:  `origin_station_id`/`dest_station_id` and `origin_station_dis`/`dest_station_dis`have no info gain against knowing the airport alone.
# MAGIC - Decision: drop all 4

# COMMAND ----------

# Group by 'ORIGIN', collect unique 'ORIGIN_AIRPORT_ID' values into a list, and count them
origin_station_ids_per_origin = otpw_1y.groupBy("ORIGIN")\
    .agg(
        F.collect_set("origin_station_id").alias("origin_station_ids"),\
            F.size(F.collect_set("origin_station_id")).alias("origin_station_id_count")\
    )\
    .orderBy("origin_station_id_count",ascending=False)

# show the result
origin_station_ids_per_origin.show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC Let's drop these, as well as `origin_station_dis` and `dest_station_dis` since they have no info gain against knowing the airport alone. 

# COMMAND ----------

# drop station cols
funcs.get_df_dimensions(otpw_1y)
drop_station_cols = ['origin_station_id','origin_station_dis','dest_station_id','dest_station_dis']
otpw_1y = otpw_1y.drop(*drop_station_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8. drop 'DIVERTED' (all values=0 since we dropped diverted flight observations) & 'DISTANCE_GROUP' (less granular than distance)

# COMMAND ----------

# drop other extraneous cols
funcs.get_df_dimensions(otpw_1y)
drop_extra_cols = [ 'DIVERTED','DISTANCE_GROUP']
otpw_1y = otpw_1y.drop(*drop_extra_cols)
funcs.get_df_dimensions(otpw_1y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Separate out "model dataset" (model features only) + "full dataset" (model + reference features )

# COMMAND ----------

otpw_1y.columns

# COMMAND ----------

# list reference columns (for feature engineering, etc., but not used directly in modeling)
reference_cols = [ # 10 columns total
    'two_hours_prior_depart_UTC', # for FE only
    'MONTH',
    'CRS_DEP_TIME',
    'DEP_TIME',
    'DEP_DELAY', 
    'CRS_ARR_TIME', # for FE only (w/ previous flights before 2 hr. window)
    'ARR_TIME', # for FE only (w/ previous flights before 2 hr. window)
    'ARR_DEL15', # can probably drop this TBH
    'CRS_ELAPSED_TIME',
    'ACTUAL_ELAPSED_TIME']

otpw_1y_full = otpw_1y # cleaned dataset (as is)
all_cols = otpw_1y_full.columns # list all columns in full dataset 
model_only_cols = [col for col in all_cols if col not in reference_cols]

# select only model columns for model version of data 
otpw_1y_model = otpw_1y.select(model_only_cols)

print("Dimensions of otpw_1y_full: ")
funcs.get_df_dimensions(otpw_1y_full)

print("Dimensions of otpw_1y_model: ")
funcs.get_df_dimensions(otpw_1y_model)

# COMMAND ----------

# let's note for next steps: which categorical/ordinal columns will need to be encoded for modeling?
string_columns = [(column_name,data_type) for column_name, data_type in otpw_1y_model.dtypes if data_type == 'string']
string_columns

# COMMAND ----------

otpw_1y_full.printSchema()

# COMMAND ----------

# write both versions to storage
# funcs.write_parquet_to_blob(otpw_1y_full, LH/clean/otpw_1y_full)
otpw_1y_full.write.mode('overwrite').parquet(f"{team_blob_url}/LH/clean/otpw_1y_full")
otpw_1y_model.write.mode('overwrite').parquet(f"{team_blob_url}/LH/clean/otpw_1y_model")
# funcs.write_parquet_to_blob(otpw_1y_model, LH/cleaned/otpw_1y_model)
# otpw_1y_full.write.mode('overwrite').parquet(f"{team_blob_url}/LH/")
# otpw_1y_model.write.mode('overwrite').parquet(f"{team_blob_url}/LH/")


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data visualization: average delays by time period

# COMMAND ----------

# Days in each month for a regular (non-leap) year
days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Calculate the 'day_of_year' for the first day of each month
day_of_year_first_of_month = []
current_day = 1  # Start from the first day of the year

for days in days_in_months:
    day_of_year_first_of_month.append(current_day)
    current_day += days  # Add the number of days in the current month for the next month's first day

#Corresponding abbreviations for each month's first day
month_abbreviations = ["1/1", "2/1", "3/1", "4/1", "5/1", "6/1", "7/1", "8/1", "9/1", "10/1", "11/1", "12/1"]

# Create a mapping of day_of_year to month abbreviation
day_to_month_map = {day: month for day, month in zip(day_of_year_first_of_month, month_abbreviations)}

# COMMAND ----------

# delays by day of year 

# Group by 'day_of_year' and calculate the average 'DELAY_TIME' for each day
result_df = otpw_1y.groupBy("day_of_year").agg(F.avg(F.col("DEP_DELAY")).alias("Average_DEP_DELAY"))

# Convert the result to a Pandas DataFrame (for plotting)
result_pd = result_df.toPandas()

# # Plotting
plt.figure(figsize=(15, 6))
sns.barplot(x=result_pd["day_of_year"], y=result_pd["Average_DEP_DELAY"])

# Update x-ticks
ax = plt.gca()
new_labels = []
for label in ax.get_xticklabels():
    day = int(label.get_text())
    new_label = day_to_month_map.get(day, '')  # Get the abbreviation or an empty string
    new_labels.append(new_label)

ax.set_xticklabels(new_labels)

# Plot the average delay time by day of the year
plt.xlabel("Day of Year")
plt.ylabel("Average Departure Delay (Minutes)")
plt.title("Average Departure Delay by Day of Year")
plt.show()


# COMMAND ----------

# delays by day of week 

# Group by 'day_of_year' and calculate the average 'DELAY_TIME' for each day
result_df = otpw_1y.groupBy("DAY_OF_WEEK").agg(F.avg(F.col("DEP_DELAY")).alias("Average_DEP_DELAY"))

# Convert the result to a Pandas DataFrame (for plotting)
result_pd = result_df.toPandas()

#Plot the average delay time by day of the year
plt.figure(figsize=(12, 6))
sns.barplot(x=result_pd["DAY_OF_WEEK"], y=result_pd["Average_DEP_DELAY"])
plt.xlabel("Day of Week")
plt.ylabel("Average Departure Delay (Minutes)")
plt.title("Average Departure Delay by Day of Week (2015)")
plt.show()

# COMMAND ----------

def_plot_average_delay_time_by_carrier(spark_df, )
#airline_names_df = airline_names_df.toPandas()

# Group by 'OP_UNIQUE_CARRIER' and calculate the average 'DEP_DELAY'
# avg_delay_df = otpw_1y.groupBy("OP_UNIQUE_CARRIER").agg(F.avg("DEP_DELAY").alias("Average_DEP_DELAY"))

# # Convert the result to a Pandas DataFrame for plotting
# avg_delay_pd = avg_delay_df.toPandas()

# # Sort the data for better visualization
#avg_delay_pd = avg_delay_pd.merge(airline_names_df, left_on='OP_UNIQUE_CARRIER', right_on='Code')

# Plotting using Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x="Airline", y="Average_DEP_DELAY", data=avg_delay_pd)
plt.xlabel("Airline Carrier")
plt.ylabel("Average Departure Delay (Minutes)")
plt.title("Average Departure Delay by Airline Carrier")
plt.xticks(rotation=45)  # Rotate the labels for better readability
plt.tight_layout()  # Adjust the layout to fit everything
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Data visualization: average delay by carrier 

# COMMAND ----------

#average delay by carrier
#airline_names_df = airline_names_df.toPandas()

# Group by 'OP_UNIQUE_CARRIER' and calculate the average 'DEP_DELAY'
# avg_delay_df = otpw_1y.groupBy("OP_UNIQUE_CARRIER").agg(F.avg("DEP_DELAY").alias("Average_DEP_DELAY"))

# # Convert the result to a Pandas DataFrame for plotting
# avg_delay_pd = avg_delay_df.toPandas()

# # Sort the data for better visualization
#avg_delay_pd = avg_delay_pd.merge(airline_names_df, left_on='OP_UNIQUE_CARRIER', right_on='Code')

# Plotting using Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x="Airline", y="Average_DEP_DELAY", data=avg_delay_pd)
plt.xlabel("Airline Carrier")
plt.ylabel("Average Departure Delay (Minutes)")
plt.title("Average Departure Delay by Airline Carrier")
plt.xticks(rotation=45)  # Rotate the labels for better readability
plt.tight_layout()  # Adjust the layout to fit everything
plt.show()

# COMMAND ----------

# Select the subset of columns you want to analyze
selected_columns = ['DEP_DELAY', 'ARR_DELAY', 'origin_HourlyPrecipitation', 'dest_HourlyPrecipitation',
                    'origin_3Hr_Precipitation', 'dest_3Hr_Precipitation', 'origin_6Hr_Precipitation',
                    'dest_6Hr_Precipitation', 'origin_12Hr_Precipitation', 'dest_12Hr_Precipitation',
                    'origin_DailyPrecipitation', 'dest_DailyPrecipitation']

selected_data = otpw_1y.select(selected_columns)

 # Convert to Pandas DataFrame
pandas_df = selected_data.toPandas()

# # Calculate the correlation matrix
corr_matrix = pandas_df.corr()
# Set the values in the upper triangle of the correlation matrix to NaN
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_matrix[mask] = np.nan


# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Precipitation &  Delay Features: Pearson Correlation Heatmap")
plt.show()

# COMMAND ----------

# all_columns = otpw_1y.columns

# # Filter columns that contain 'Temperature'
#temperature_columns = [col for col in all_columns if 'Temperature' in col]

temperature_columns = ['DEP_DELAY','ARR_DELAY','origin_DailyDepartureFromNormalAverageTemperature', 'origin_DailyAverageDryBulbTemperature', 'dest_DailyDepartureFromNormalAverageTemperature', 'dest_DailyAverageDryBulbTemperature', 'origin_HourlyDryBulbTemperature', 'origin_3Hr_DryBulbTemperature', 'origin_6Hr_DryBulbTemperature', 'origin_12Hr_DryBulbTemperature', 'dest_HourlyDryBulbTemperature', 'dest_3Hr_DryBulbTemperature', 'dest_6Hr_DryBulbTemperature', 'dest_12Hr_DryBulbTemperature']

# # Select only the filtered columns
df_filtered = otpw_1y.select(temperature_columns)

pandas_df = df_filtered.toPandas()

# # Calculate the correlation matrix
corr_matrix = pandas_df.corr()
# Set the values in the upper triangle of the correlation matrix to NaN
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_matrix[mask] = np.nan


# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Temperature Delay Features: Pearson Correlation Heatmap")
plt.show()



# COMMAND ----------

# def make_feature_subset_heatmap(spark_df, str_to_select_cols,plot_title):
#     # list all columns in spark df
#     all_columns = spark_df.columns
#     # select columns containing string that identifies the feature subset
#     heatmap_columns = [col for col in all_columns if str_to_select_cols in col]
#     # subset those columns in spark df
#     df_filtered = otpw_1y.select(pressure_columns)
#     # convert to pandas df
#     pandas_df = df_filtered.toPandas()
#     # calculate the correlation matrix
#     corr_matrix = pandas_df.corr()
#     # set repeated values in upper triangle of correlation matrix to NaN (simplify plot)
#     mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#     corr_matrix[mask] = np.nan
#     #Plot the heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
#     plt.title(plot_title)
#     plt.show()


# COMMAND ----------


make_feature_subset_pearson_corr_heatmap(spark_df, str_to_select_cols,plot_title)

# Filter columns that contain 'pressure'
pressure_columns = [col for col in all_columns if 'Pressure' in col]
# # Select only the filtered columns
df_filtered = otpw_1y.select(pressure_columns)

pandas_df = df_filtered.toPandas()

# # Calculate the correlation matrix
corr_matrix = pandas_df.corr()
# Set the values in the upper triangle of the correlation matrix to NaN
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_matrix[mask] = np.nan


# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Pressure & Delay Features: Pearson Correlation Heatmap")
plt.show()


# COMMAND ----------

# Filter columns that contain 'Temperature'
pressure_columns = [col for col in all_columns if 'Visibility' in col]
pressure_columns = ['DEP_DELAY','ARR_DELAY','origin_HourlyVisibility', 'origin_3Hr_Visibility', 'origin_6Hr_Visibility', 'origin_12Hr_Visibility', 'dest_HourlyVisibility', 'dest_3Hr_Visibility', 'dest_6Hr_Visibility', 'dest_12Hr_Visibility']

# # Select only the filtered columns
df_filtered = otpw_1y.select(pressure_columns)

pandas_df = df_filtered.toPandas()

# # Calculate the correlation matrix
corr_matrix = pandas_df.corr()
# Set the values in the upper triangle of the correlation matrix to NaN
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_matrix[mask] = np.nan

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Visibility Delay Features: Pearson Correlation Heatmap")
plt.show()


# COMMAND ----------

# flights per route
 
#  Selecting distinct combinations of origin & destination airports 
distinct_combinations= otpw_1y.select('ORIGIN','DEST','OP_CARRIER_FL_NUM').distinct()
# distinct_combinations
# # distinct_combinations.groupBy('ORIGIN','DEST')
distinct_combinations = distinct_combinations.toPandas()
distinct_combinations.sort_values(by="OP_CARRIER_FL_NUM",ascending=FALSE)

# COMMAND ----------

# Selecting distinct combinations of origin & destination airports 
distinct_combinations= otpw_1y.select('OP_CARRIER_FL_NUM','ORIGIN','DEST','FL_DATE').distinct()
distinct_combinations = distinct_combinations.groupBy('ORIGIN','DEST','FL_DATE').count()
display(distinct_combinations)

# COMMAND ----------

# Step 1: Calculate the counts
route_counts = otpw_1y.groupBy('ORIGIN', 'DEST', 'FL_DATE') \
                      .count() \
                      .withColumnRenamed('count', 'n_flights_per_route_by_day')

# Step 2: Join back to the original DataFrame
otpw_1y = otpw_1y.join(route_counts, ['ORIGIN', 'DEST', 'FL_DATE'])

# COMMAND ----------

feats = ["n_flights_per_route_by_day","DEP_DELAY"]
funcs.pairplot(otpw_1y,feats)

# COMMAND ----------

# Group by 'ORIGIN' and 'DEST', and count distinct 'OP_UNIQUE_CARRIER' values
distinct_carriers = otpw_1y.groupBy('ORIGIN', 'DEST') \
                           .agg(F.countDistinct('OP_UNIQUE_CARRIER').alias('distinct_carriers'))

# Show the result
distinct_carriers.show()
# carriers_per_route_counts = otpw_1y.groupBy('ORIGIN', 'DEST', 'OP_UNIQUE_CARRIER') \
#                       .count() \
#                       .withColumnRenamed('count', 'n_carriers_per_route')
# display(carriers_per_route_counts)
# # # Step 2: Join back to the original DataFrame
# # otpw_1y = otpw_1y.join(route_counts, ['ORIGIN', 'DEST'])

# COMMAND ----------

# Step 1: Calculate the count of distinct carriers per route
distinct_carriers_count = otpw_1y.groupBy('ORIGIN', 'DEST') \
                                 .agg(F.countDistinct('OP_UNIQUE_CARRIER').alias('n_carriers_per_route'))

# Step 2: Join this back to the original DataFrame
otpw_1y = otpw_1y.join(distinct_carriers_count, on=['ORIGIN', 'DEST'])


# COMMAND ----------

feats = ["n_carriers_per_route","DEP_DELAY"]
funcs.pairplot(otpw_1y,feats,sample_size=10000)

# COMMAND ----------

# average delay by origin



average_delay_by_origin = otpw_1y.groupBy("ORIGIN").agg(F.avg("DEP_DELAY").alias("avg_dep_delay"))
average_delay_by_origin = average_delay_by_origin.toPandas()
average_delay_by_origin= average_delay_by_origin.sort_values(by='avg_dep_delay',ascending=False)
# plt.bar(average_delay_by_origin, height='avg_dep_delay')
# Display top N airports
N = 20
top_airports = average_delay_by_origin.head(N)

plt.figure(figsize=(10, 6))
sns.barplot(x='avg_dep_delay', y='ORIGIN', data=top_airports)
plt.xlabel('Average Departure Delay (minutes)')
plt.ylabel('Origin Airport')
plt.title(f'Top {N} Airports with Highest Average Departure Delay')
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC TO DO: general visualizations 
# MAGIC - 'TAIL_NUM'	Plan			
# MAGIC - 'OP_CARRIER_FL_NUM'	Flight			
# MAGIC - 'DISTANCE'	Flight			

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC TO DO: add Art's engineered features 
# MAGIC - 'Av_airport_delay'	Airport	average delay for an airport for 4 hour window, at 2 hours before flight
# MAGIC - 'Prev_delay'	Plane	Departure delay from last flight with same tail number (flights departing at least two hours prior)
# MAGIC - 'Av_carrier_delay'	Airline	average delay for a carrier for 4 hour window, at 2 hours before flight
# MAGIC - 'Num_flights'	Airline	number of flights used for above carrier calculation
# MAGIC - 'sq_precip'	Precipitation	squared value of 'origin_3Hr_Precipitation'
# MAGIC - 'sq_snow'	Precipitation	squared value of 'origin_DailySnowfall'
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes/ideas on future feature engineering: 
# MAGIC - airport pairs 
# MAGIC - carriers per route
# MAGIC - arr delay prev flight segment 
# MAGIC - delays/weather @ origin of arriving flights
# MAGIC - avg arr delay @ origin airport 2 hours before takeoff
# MAGIC - avg dep delay @ arriving flights' origin 2 hours before takeoff 

# COMMAND ----------


