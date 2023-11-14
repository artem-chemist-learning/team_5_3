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
from pyspark.sql.functions import size, to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when, collect_list, udf, date_trunc

# COMMAND ----------

# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"
display(dbutils.fs.ls(f"{mids261_mount_path}"))


# COMMAND ----------

# read data from file
otpw = spark.read.load(f"{mids261_mount_path}/OTPW_3M_2015.csv",format="csv", inferSchema="true", header="true")

# COMMAND ----------

#take only columns needed
otpw_to_process = otpw[['sched_depart_date_time_UTC','DEP_DELAY', 'ORIGIN', 'TAIL_NUM']].dropna()

#convert date and time to datetime format and drop original (now unneeded) column
#otpw_to_process = otpw_to_process.withColumn('datetime', to_timestamp(otpw_to_process['sched_depart_date_time_UTC'])).drop('sched_depart_date_time_UTC')
otpw_to_process = otpw_to_process.withColumn('date_day', date_trunc('day', otpw_to_process.sched_depart_date_time_UTC)).orderBy('sched_depart_date_time_UTC')

# Show top data for LAX
otpw_to_process.filter(otpw_to_process.ORIGIN == 'LAX').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at the data at the Airport/Day level

# COMMAND ----------

# Define a function that takes a list, converts it into pd dataframe and uses Pandas built-in function to compute autocorrelation with lag 1
def autocorr(ret):
  s = pd.Series(ret)
  return float(s.autocorr(lag=1))

auto = udf(autocorr, FloatType())

# Group by Origin and day
# Then compute autocorrelation with lag 1 for the resulting groups: for each airport, each day
# Also keep track of how many flights airport has seen that day
df_auto_origin = otpw_to_process.groupBy("ORIGIN","date_day") \
                .agg(collect_list(col("DEP_DELAY")).alias("DEP_DELAYS")) \
                .withColumns({"auto": auto("DEP_DELAYS"), "count": size("DEP_DELAYS")})\
                .orderBy("date_day")

# Discard records for the airports with less than 5 flights a day.
df_auto_origin = df_auto_origin.filter(df_auto_origin["count"]>4)[['ORIGIN', 'date_day', 'auto']]

#Get final result into Pandas. Only ~20k records for 3M data
pd_auto_origin = df_auto_origin.toPandas()
pd_auto_origin


# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at the Tailnumber/Day level

# COMMAND ----------

# Group by TailNUmber and day
# Then compute autocorrelation with lag 1 for the resulting groups: for each airport, each day
# Also keep track of how many flights airport has seen that day
df_auto_tail = otpw_to_process.groupBy("TAIL_NUM","date_day") \
                .agg(collect_list(col("DEP_DELAY")).alias("DEP_DELAYS")) \
                .withColumns({"auto": auto("DEP_DELAYS"), "count": size("DEP_DELAYS")})\
                .orderBy("date_day")

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


