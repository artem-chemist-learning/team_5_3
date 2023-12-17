# Databricks notebook source
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
# load clean 5-yr train data
df = spark.read.parquet(f"{team_blob_url}/BK/clean_5yr_WITHOUT_2019_eng")

# COMMAND ----------

df.columns

# COMMAND ----------

from pyspark.sql.functions import avg

# Calculate the average airport delay based on isHolidayWindow
avg_delay = df.groupBy('isHolidayWindow').agg(avg('Av_airport_delay').alias('avg_airport_delay'))

# Convert the Spark DataFrame to Pandas for plotting
avg_delay_pandas = avg_delay.toPandas()

# Plotting the data
plt.bar(avg_delay_pandas['isHolidayWindow'], avg_delay_pandas['avg_airport_delay'], color='skyblue')
plt.xlabel('isHolidayWindow')
plt.ylabel('Average Airport Delay')
plt.title('Average Airport Delay vs isHolidayWindow')
plt.xticks(avg_delay_pandas['isHolidayWindow'])  # Set x-axis ticks to isHolidayWindow values
plt.show()


# COMMAND ----------

# list generated from ChatGPT
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import size, to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when, collect_list, udf, date_trunc, count, lag, first, last, percent_rank, array, year, month, dayofmonth


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
df = df.withColumn('isHolidayWindow',
    is_holiday_window_extended_udf(col('day_of_year'))
)

# COMMAND ----------

# Assuming 'day_of_year' ranges from 1 to 366
# Calculate the average delay for each day, considering whether it's a holiday window or not
avg_delay_per_day = df.groupBy('day_of_year', 'isHolidayWindow').agg(avg('airport_average_hourly').alias('avg_delay'))

# Convert the Spark DataFrame to Pandas for plotting
avg_delay_per_day_pandas = avg_delay_per_day.toPandas()

# Check the data in Pandas DataFrame (for debugging)
print(avg_delay_per_day_pandas.head())  # Check the first few rows of the DataFrame

# Separate data for holidays and non-holidays
holiday_data = avg_delay_per_day_pandas[avg_delay_per_day_pandas['isHolidayWindow'] == 1.0]
non_holiday_data = avg_delay_per_day_pandas[avg_delay_per_day_pandas['isHolidayWindow'] == 0.0]

# Plotting the data
plt.figure(figsize=(10, 6))

# Plot average delay for non-holidays
plt.scatter(non_holiday_data['day_of_year'], non_holiday_data['avg_delay'], label='Non-Holiday', color='blue')

# Plot average delay for holidays
plt.scatter(holiday_data['day_of_year'], holiday_data['avg_delay'], label='Holiday', color='red', marker='x')


plt.xlabel('Day of the Year')
plt.ylabel('Average Airport Delay')
plt.title('Average Airport Delay vs Day of the Year with Holidays')
plt.legend()
#plt.grid(True)
plt.xlim(1, 366)  # Set x-axis limits from 1 to 366
plt.ylim(0, None)  # Set y-axis lower limit to 0 (upper limit will be determined automatically)
plt.show()


# COMMAND ----------

plt.xlabel('Day of the Year')
plt.ylabel('Average Airport Delay')
plt.title('Average Airport Delay vs Day of the Year with Holidays')
plt.legend()
#plt.grid(True)
plt.xlim(1, 366)  # Set x-axis limits from 1 to 366
plt.ylim(0, None)  # Set y-axis lower limit to 0 (upper limit will be determined automatically)
plt.show()

plt.scatter(avg_delay_per_day_pandas['day_of_year'],avg_delay_per_day_pandas['avg_delay'])

# COMMAND ----------

# Assuming 'day_of_year' ranges from 1 to 366
# Calculate the average delay for each day, considering whether it's a holiday window or not
avg_delay_per_day = df.groupBy('day_of_year', 'isHolidayWindow').agg(avg('Av_airport_delay').alias('avg_delay'))

# Convert the Spark DataFrame to Pandas for plotting
avg_delay_per_day_pandas = avg_delay_per_day.toPandas()

# Plotting the data
plt.figure(figsize=(10, 6))

# Plotting average delay for all days, not explicitly differentiating between holidays and non-holidays
plt.scatter(avg_delay_per_day_pandas['day_of_year'], avg_delay_per_day_pandas['avg_delay'], color='green', marker='o')

plt.xlabel('Day of the Year')
plt.ylabel('Average Airport Delay')
plt.title('Average Airport Delay vs Day of the Year')
plt.grid(True)
plt.xlim(1, 366)  # Set x-axis limits from 1 to 366
plt.ylim(0, None)  # Set y-axis lower limit to 0 (upper limit will be determined automatically)
plt.show()


# COMMAND ----------

# Assuming 'day_of_year' ranges from 1 to 366
# Calculate the average delay for each day, considering whether it's a holiday window or not
avg_delay_per_day = df.groupBy('day_of_year', 'isHolidayWindow').agg(avg('Av_airport_delay').alias('avg_delay'))

# Convert the Spark DataFrame to Pandas for plotting
avg_delay_per_day_pandas = avg_delay_per_day.toPandas()

# Plotting the data with different colors for holiday and non-holiday points
plt.figure(figsize=(10, 6))

# Assigning colors based on 'isHolidayWindow' column values
colors = ['red' if x == 1 else 'blue' for x in avg_delay_per_day_pandas['isHolidayWindow']]

# Plotting average delay with colors based on holiday or non-holiday
plt.scatter(avg_delay_per_day_pandas['day_of_year'], avg_delay_per_day_pandas['avg_delay'], color=colors, marker='o')

# Adding legend for clarity
plt.scatter([], [], color='red', label='Holiday')  # Empty scatter plot for red (Holiday)
plt.scatter([], [], color='blue', label='Non-Holiday')  # Empty scatter plot for blue (Non-Holiday)
plt.xlabel('Day of the Year')
plt.ylabel('Average Airport Delay')
plt.title('Average Airport Delay vs Day of the Year')
plt.legend()
plt.grid(True)
plt.xlim(1, 366)  # Set x-axis limits from 1 to 366
plt.ylim(0, None)  # Set y-axis lower limit to 0 (upper limit will be determined automatically)
plt.show()


# COMMAND ----------


