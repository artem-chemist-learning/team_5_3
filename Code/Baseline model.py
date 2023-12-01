# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Technical notebook to create statistical baseline model

# COMMAND ----------

# importing custom functions
from Code.funcs import blob_connect, write_parquet_to_blob
import csv
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, FloatType, DoubleType,  ArrayType
from pyspark.sql.functions import size, to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when, collect_list, udf, date_trunc, count, lag, first, last, percent_rank, array
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, IndexToString
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression as LR

team_blob_url = blob_connect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read from storage

# COMMAND ----------

# read in daily weather data from parquet
joined3M = spark.read.parquet(f"{team_blob_url}/LH/1yr_clean_temp_2")

# COMMAND ----------

categorical_features = ['ORIGIN']

aux_features = ['sched_depart_date_time_UTC', 'TAIL_NUM', 'DEP_DELAY']

# COMMAND ----------

# take only columns with simple features for this model
# FIlter out all cancelled
df_clean = joined3M[aux_features + categorical_features].dropna()


# COMMAND ----------

# Make a binary column 1 is for delay
df_clean = df_clean.withColumn('label', (df_clean.DEP_DELAY >=15).cast('integer'))

# COMMAND ----------

df_clean.dtypes

# COMMAND ----------

# Make new column with time in seconds since the begining of Unix epoch
df_clean = df_clean.withColumn('time_long', df_clean.sched_depart_date_time_UTC.cast("long")).orderBy(df_clean.sched_depart_date_time_UTC)

#Helper function to navigate this column
hours = lambda i: i * 3600

# average delay for this airport
Time_Origin_Window = Window.partitionBy('ORIGIN').orderBy(col('time_long')).rangeBetween(-hours(6), -hours(2))
df_clean = df_clean.withColumns({"predicted_delay": _mean("DEP_DELAY").over(Time_Origin_Window)}).fillna(0)

# Calculate errors for each flight
df_clean = df_clean.withColumn('error', df_clean.predicted_delay - df_clean.DEP_DELAY)

# COMMAND ----------

# Get the stats for the predictions

df_error_stats = df_clean.select(
    _mean(col('error')).alias('mean_error'),
    _stddev(col('error')).alias('std_error')
).collect()

mean_error = df_error_stats[0]['mean_error']
std_error = df_error_stats[0]['std_error']

print(f'Mean error: {round(mean_error, 2)}, Error stddev: {round(std_error, 2)}')


# COMMAND ----------

#Get number of delated and total number of flights
pos_stats = df_clean.select(
    _sum(col('label')).alias('num_delayed')
    ).collect()

Positive = pos_stats[0]['num_delayed']
Total = df_clean.count()
print(f"Actually delayed: {Positive}, Total flights:{Total}")


# COMMAND ----------

# Set decison cut offs
CutOffs = [-3, 0, 2, 3, 5, 10, 30, 60]

# Define functions to labeling a prediction as FP(TP) 
# Based on teh cut off
def TP(prob_pos, label):
    return [ 1 if (prob_pos >= cut_off) and (label > 0)  else 0 for cut_off in CutOffs]
def FP(prob_pos, label):
    return [ 1 if (prob_pos >= cut_off) and (label < 1)  else 0 for cut_off in CutOffs]

# Define udfs based on these functions
# These udfs return arrays of the same length as the cut-off array
# With 1 if the decision would be TP(FP) at this cut off
make_TP = udf(TP,  ArrayType(IntegerType()))
make_FP = udf(FP,  ArrayType(IntegerType()))

# Generate these arrays in the dataframe returned by prediction
predictions = df_clean.withColumns({'TP':make_TP(df_clean.predicted_delay, df_clean.label), 'FP':make_FP( df_clean.predicted_delay,  df_clean.label)})

# Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
num_cols = len(CutOffs)
TP_FP_pd = predictions.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                        array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                        _sum(col("label")).alias("Positives")
                        )\
                        .toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
results_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
results_pd['Precision'] = 100*results_pd['TP']/(results_pd['TP'] + results_pd['FP'])
results_pd['Recall']= 100*results_pd['TP']/TP_FP_pd.iloc[0,2]
results_pd.to_csv('../Data/Average_in_airport_prec_rec.csv')
results_pd

# COMMAND ----------

# Make random predictions
# Set decison cut offs
CutOffs = [0, 0.20, 0.40, 0.60, 0.80, 0.99]

# Define functions to labeling a prediction as FP(TP) 
# Based on teh cut off
def TP(prob_pos, label):
    return [ 1 if (prob_pos >= cut_off) and (label > 0)  else 0 for cut_off in CutOffs]
def FP(prob_pos, label):
    return [ 1 if (prob_pos >= cut_off) and (label < 1)  else 0 for cut_off in CutOffs]

# Define udfs based on these functions
# These udfs return arrays of the same length as the cut-off array
# With 1 if the decision would be TP(FP) at this cut off
make_TP = udf(TP,  ArrayType(IntegerType()))
make_FP = udf(FP,  ArrayType(IntegerType()))

# Generate these arrays in the dataframe returned by prediction
df_clean = df_clean.withColumn('rnd_pred', rand(seed = 42) )
predictions = df_clean.withColumns({'TP':make_TP(df_clean.rnd_pred, df_clean.label), 'FP':make_FP( df_clean.rnd_pred,  df_clean.label)})

# Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
num_cols = len(CutOffs)

TP_FP_pd = predictions.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                        array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                        _sum(col("label")).alias("Positives")
                        )\
                        .toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
results_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
results_pd['Precision'] = 100*results_pd['TP']/(results_pd['TP'] + results_pd['FP'])
results_pd['Recall']= 100*results_pd['TP']/TP_FP_pd.iloc[0,2]
results_pd.to_csv('../Data/Random_prec_rec.csv')
results_pd

# COMMAND ----------


