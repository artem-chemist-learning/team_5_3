# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Technical notebook to create statistical baseline model

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyspark.sql.window import Window
from pyspark.sql.functions import to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when

# COMMAND ----------

class Flight:

    def __init__(self, fl_num, t_num, dep_time):

        self.flight_num = fl_num
        self.time_dep = dep_time
        self.tail_num = t_num

# COMMAND ----------

class Baseline_Model:

    def __init__(self, fl_num, t_num, dep_time):
        self.conf_level = 0.95
        self.time_dep = dep_time
        self.tail_num = t_num
    
    def fit(self, data):
        pass
    
    def predict(self, flight):
        pass

# COMMAND ----------

# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"
display(dbutils.fs.ls(f"{mids261_mount_path}"))


# COMMAND ----------

# read data from file
otpw = spark.read.format("csv").option("header","true").load(f"{mids261_mount_path}/OTPW_3M/")

#take only columns needed
otpw_to_process = otpw[['sched_depart_date_time_UTC','DEP_DELAY', 'ORIGIN']].dropna()

#convert dat and time to datetime format and drop original (now unneeded) column
otpw_to_process = otpw_to_process.withColumn('datetime', to_timestamp(otpw_to_process['sched_depart_date_time_UTC'])).drop('sched_depart_date_time_UTC')

# create window by casting timestamp to long (number of seconds)
# Window will partition by the airport
# go for 4 hours and stop just before the flight in question
hours = lambda i: i * 3600
WindowSpec = (Window.partitionBy('ORIGIN').orderBy(col('datetime').cast('long')).rangeBetween(-hours(4), -hours(2)))

# Calculate average delay over that window
otpw_to_process = otpw_to_process.withColumn('predicted_delay', _mean("DEP_DELAY").over(WindowSpec)).dropna()

# Calculate errors for each flight
otpw_to_process = otpw_to_process.withColumn('error', otpw_to_process.predicted_delay - otpw_to_process.DEP_DELAY)

# Make a binary column 1 is for delay
otpw_to_process = otpw_to_process.withColumn('delayed', (otpw_to_process.DEP_DELAY >=15))

# Show top data for LAX
otpw_to_process.filter(otpw_to_process.ORIGIN == 'LAX').orderBy('datetime').show()

# COMMAND ----------

# Ge the stats for the predictions

df_error_stats = otpw_to_process.select(
    _mean(col('error')).alias('mean_error'),
    _stddev(col('error')).alias('std_error')
).collect()

mean_error = df_error_stats[0]['mean_error']
std_error = df_error_stats[0]['std_error']

print(f'Mean error: {round(mean_error, 2)}, Error stddev: {round(std_error, 2)}')


# COMMAND ----------

#Get number of delated and total number of flights
pos_stats = otpw_to_process.select(
    _sum(col('delayed').cast('integer')).alias('num_delayed')
    ).collect()

Positive = pos_stats[0]['num_delayed']
Total = otpw_to_process.count()
print(f"Actually delayed: {Positive}, Total flights:{Total}")


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

# Make random preditions, given probability of prediction
def random_prec_rec(prob, num_positive, data):

    data = data.withColumn('rnd_pred', when(rand(seed = 42) > prob, True).otherwise(False))

    df_pred_stats = data.select(
        _sum(  (data.rnd_pred & data.delayed    ).cast('integer') ).alias('TP'),
        _sum(  (data.rnd_pred & ( ~data.delayed) ).cast('integer') ).alias('FP')
    ).collect()

    TP = df_pred_stats[0]['TP']
    FP = df_pred_stats[0]['FP']

    precision = 100*TP/(TP+FP)
    recall = 100*TP/num_positive

    return precision, recall
results = []
probs =  [0, 0.20, 0.40, 0.60, 0.80, 0.99]
for i in probs:
    results.append(random_prec_rec(i, Positive, otpw_to_process))

results_pd_rnd = pd.DataFrame(results)
results_pd_rnd.columns = ['Precision', 'Recall']
results_pd_rnd

# COMMAND ----------

# MAGIC %md
# MAGIC #Make precision/recall graphs for two models

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
axes.plot(results_pd_rnd.Recall, results_pd_rnd.Precision, label = "Random", color = 'r') 
axes.scatter (results_pd_rnd.Recall, results_pd_rnd.Precision, label = "Probability", color = 'r') 


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


