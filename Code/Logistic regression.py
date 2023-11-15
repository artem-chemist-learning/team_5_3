# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Technical notebook to explore autocorrelations in the timesries data

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import size, to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when, collect_list, udf, date_trunc, count, lag, first, last

# COMMAND ----------

# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"
# read data from file
otpw = spark.read.load(f"{mids261_mount_path}/OTPW_3M_2015.csv",format="csv", inferSchema="true", header="true")

# COMMAND ----------

# Let's calculate percentage of nulls for each field, given the nulls and count of each field
data_size = int(otpw.count())
null_percents = otpw.select([(100.0 * count(when(col(c).isNull(), c))/data_size).alias(c) for c in otpw.columns])

# Filtering out columns where there were more than 90% of the data missing
null_per_t = null_percents.toPandas().T.reset_index(drop=False)
null_per_t = null_per_t[null_per_t[0] > 90]

# Lastly, we will add cols have too many nulls to the list of columns to drop
drop_cols = null_per_t['index'].tolist()

# Remove columns that have perfectly correlated duplicates
drop_cols = drop_cols +['ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'NAME', 'origin_airport_name', 'dest_airport_name', 'ORIGIN_STATE_NM', 'DEST_STATE_NM']

# Drop unneeded columns and cancelled flights.
otpw = otpw.drop(*drop_cols).filter(otpw.CANCELLED < 0.1).drop('CANCELLED')

# COMMAND ----------

hourly_weather_features = ['HourlyDryBulbTemperature',
            'HourlyStationPressure',
            'HourlyPressureChange',
            'HourlyWindGustSpeed',
            'HourlyWindDirection',
            'HourlyPrecipitation',
            'HourlyVisibility',
]

# casting daily features as float
for col_name in hourly_weather_features:
    otpw = otpw.withColumn(col_name, col(col_name).cast('float'))

# removing trace values
# otpw[hourly_weather_features] = otpw[hourly_weather_features].replace('T', '0.005')

# COMMAND ----------

# Take only data needed for toy LR
#take only columns needed
df_clean = otpw[['sched_depart_date_time_UTC','DEP_DELAY', 'ORIGIN', 'HourlyPrecipitation', 'TAIL_NUM']].dropna()

# COMMAND ----------

df_clean = df_clean.withColumn('time_long', df_clean.sched_depart_date_time_UTC.astype('Timestamp').cast("long")).orderBy(df_clean.sched_depart_date_time_UTC)

# COMMAND ----------

df_clean.orderBy(df_clean.sched_depart_date_time_UTC).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### New feature: mean precip over 2h period from 4 hours before the flight to 2 hours before the flight  

# COMMAND ----------

# create window by casting timestamp to long (number of seconds)
# Window will partition by the airport
# go for 4 hours and stop just before the flight in question
hours = lambda i: i * 3600
Time_Origin_Window = Window.partitionBy('ORIGIN').orderBy(col('time_long')).rangeBetween(-hours(4), -hours(2))

# Calculate HourlyPrecipitation over that window
df_clean = df_clean.withColumn('precip_2h', _mean('HourlyPrecipitation').over(Time_Origin_Window)).dropna()
df_clean.orderBy('ORIGIN', 'time_long').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### New feature: previos delay for this tail number  

# COMMAND ----------

Time_Tail_Window = Window.partitionBy('TAIL_NUM').orderBy(col('time_long')).rangeBetween(-hours(20), -hours(2))

df_clean = df_clean.withColumn("Prev_delay", last("DEP_DELAY").over(Time_Tail_Window)).fillna(0)
df_clean.orderBy('TAIL_NUM', 'time_long').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at the Tail Number/Day level

# COMMAND ----------

# Group by TailNUmber and day
# Then compute autocorrelation with lag 1 for the resulting groups: for each airport, each day
# Also keep track of how many flights airport has seen that day
df_auto_tail = otpw_to_process.groupBy("TAIL_NUM","date_day") \
                .agg(collect_list(col("DEP_DELAY")).alias("DEP_DELAYS")) \
                .withColumns({"auto": auto("DEP_DELAYS"), "count": size("DEP_DELAYS")})\
                .orderBy("sched_depart_date_time_UTC")

# Discard records for the airports with less than 5 flights a day.
df_auto_tail = df_auto_tail.filter(df_auto_tail["count"]>4)[["TAIL_NUM", 'date_day', 'auto']]

#Get final result into Pandas. Only ~20k records for 3M data
pd_auto_tail = df_auto_tail.toPandas()
pd_auto_tail


# COMMAND ----------

# MAGIC %md
# MAGIC https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Make graphs for two parameters

# COMMAND ----------

# Instantiate figure and axis
num_rows = 1
num_columns = 1
fig, axes = plt.subplots(num_rows, num_columns, sharex=True)
fig.set_figheight(5)
#Adjust space between plots in the figure
plt.subplots_adjust(hspace = 0.2)


num_bins = 100

# the histogram of the data
n, bins, patches = axes.hist(pd_auto_tail.auto, num_bins, density=True)
sigma = pd_auto_tail.auto.std()
mu = pd_auto_tail.auto.mean()
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
axes.plot(bins, y, '--')

# add a 'best fit' line
axes.set_xlabel('Value')
axes.set_ylabel('Probability density')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()

#Set title and axis legend, only set axis legend on the sides
axes.legend(loc = 'upper left')
#axes.set_ylim(5, 70)


# Remove the bounding box to make the graphs look less cluttered
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
plt.show()
fig.savefig(f"Hist for autocorr.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------

# Instantiate figure and axis
num_rows = 1
num_columns = 1
fig, axes = plt.subplots(num_rows, num_columns, sharex=True)
fig.set_figheight(5)
#Adjust space between plots in the figure
# plt.subplots_adjust(hspace = 0.2)

num_bins = 100

# the histogram of the data
n, bins, patches = axes.hist(pd_auto_origin.auto, num_bins, density=True)
sigma = pd_auto_origin.auto.std()
mu = pd_auto_origin.auto.mean()
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
axes.plot(bins, y, '--')

# add a 'best fit' line
axes.set_xlabel('Value')
axes.set_ylabel('Probability density')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()

#Set title and axis legend, only set axis legend on the sides
axes.legend(loc = 'upper left')
#axes.set_ylim(5, 70)

# Remove the bounding box to make the graphs look less cluttered
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
plt.show()
fig.savefig(f"Hist for autocorr.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------


