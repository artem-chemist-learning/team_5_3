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
from pyspark.sql.functions import size, to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when, collect_list, udf, date_trunc, count, lag, first, last, percent_rank
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read data from csv, infer Schema along the way

# COMMAND ----------

# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"
# read data from file
otpw = spark.read.load(f"{mids261_mount_path}/OTPW_3M_2015.csv",format="csv", inferSchema="true", header="true")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop empty columns, repetitve columns and cancelled flights

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

# MAGIC %md
# MAGIC ### Take only features needed, fix datatypes, make tiem in seconds column

# COMMAND ----------

#Set the dataypes for the importnat hourly weather columns

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
df_clean = otpw[['sched_depart_date_time_UTC','DEP_DELAY', 'ORIGIN', 'HourlyPrecipitation', 'TAIL_NUM','OP_UNIQUE_CARRIER']].dropna()

# Make new column with time in seconds since the begining of Unix epoch
df_clean = df_clean.withColumn('time_long', df_clean.sched_depart_date_time_UTC.astype('Timestamp').cast("long")).orderBy(df_clean.sched_depart_date_time_UTC)

# COMMAND ----------

# MAGIC %md
# MAGIC ### New feature: mean precip over 2h period from 4 hours before the flight to 2 hours before the flight  

# COMMAND ----------

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
# MAGIC ### New feature: sparse vector for the airline

# COMMAND ----------

indexer = StringIndexer(inputCol="OP_UNIQUE_CARRIER", outputCol="carrier_idx")
inputs = [indexer.getOutputCol()]
encoder = OneHotEncoder(inputCols=inputs, outputCols=["carrier_vec"])
pipeline = Pipeline(stages=[indexer, encoder])
df_clean = pipeline.fit(df_clean).transform(df_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assemble all features in one vector

# COMMAND ----------

assembler = VectorAssembler().setInputCols(['carrier_vec', 'precip_2h', 'Prev_delay'] ).setOutputCol('feat_vec')
assembler_df = assembler.transform(df_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Labels

# COMMAND ----------

# Make label column
assembler_df = df_clean.withColumn('IsDelayed',  when(col("DEP_DELAY") >=15, 'Delayed').otherwise('On time'))
lbl_indexer = StringIndexer().setInputCol('IsDelayed').setOutputCol('label')

assembler_df = lbl_indexer.fit(assembler_df).transform(assembler_df)
assembler_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split

# COMMAND ----------

Rank_Window = Window.partitionBy().orderBy("sched_depart_date_time_UTC")
assembler_df = assembler_df.withColumn("rank", percent_rank().over(Rank_Window))
train_df = assembler_df.where("rank <= .8").drop('rank', 'time_long','IsDelayed')
test_df = assembler_df.where("rank > .8").drop('rank', 'time_long','IsDelayed')

# COMMAND ----------

# MAGIC %md
# MAGIC https://medium.com/swlh/logistic-regression-with-pyspark-60295d41221

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make predictions at various thresholds of what delay odds are considered a predicted delay.

# COMMAND ----------

# Make preditions, given cutoff
def prec_rec(cut_off, num_positive, data):
    df_pred_stats = data.select(
        _sum(  ((data.predicted_delay >= cut_off)  &  data.delayed  ).cast('integer')   ).alias('TP'),
        _sum(  ((data.predicted_delay >= cut_off)  &  ~data.delayed ).cast('integer')   ).alias('FP')
    ).collect()

    TP = df_pred_stats[0]['TP']
    FP = df_pred_stats[0]['FP']

    precision = 100*TP/(TP+FP)
    recall = 100*TP/num_positive

    return precision, recall
results = []
cut_offs = [-3, 0, 2, 3, 5, 10, 30, 60]
for i in cut_offs:
    results.append(prec_rec(i, Positive, otpw_to_process))

results_pd = pd.DataFrame(results)
results_pd.columns = ['Precision', 'Recall']
results_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make precision/recall graphs for two models

# COMMAND ----------

# Instantiate figure and axis
num_rows = 1
num_columns = 1
fig, axes = plt.subplots(num_rows, num_columns, sharex=True)
fig.set_figheight(10)
#Adjust space between plots in the figure
plt.subplots_adjust(hspace = 0.2)


#Fill the axis with data
axes.plot(results_pd.Recall, results_pd.Precision, label = "Previous flights", color = 'g')
axes.scatter(results_pd.Recall, results_pd.Precision,  label = "Cut off, min", color = 'g')   

axes.axvline(x=80, ymin=0.05, ymax=0.65, color='b', ls = '--')
axes.text(70, 50, '80% Recall', size=12)

#Set title and axis legend, only set axis legend on the sides
axes.legend(loc = 'upper left')

#axes[0].set_ylabel('Precision')
axes.set_ylabel('Precision')
axes.set_xlabel('Recall')
axes.set_ylim(5, 70)

for index in range(len(cut_offs)):
  axes.text(results_pd.Recall[index]-0.02, 1 + results_pd.Precision[index], cut_offs[index], size=12)
for index in range(len(probs)):
  axes.text(results_pd_rnd.Recall[index]-0.02, 1 + results_pd_rnd.Precision[index], probs[index], size=12)



# Remove the bounding box to make the graphs look less cluttered
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
plt.show()
fig.savefig(f"Precision and recall.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------


