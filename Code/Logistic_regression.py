# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Technical notebook to explore autocorrelations in the timesries data

# COMMAND ----------

# importing custom functions
from Code.funcs import blob_connect

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, FloatType, DoubleType,  ArrayType
from pyspark.sql.functions import size, to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when, collect_list, udf, date_trunc, count, lag, first, last, percent_rank, array
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression as LR

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read from storage

# COMMAND ----------

# read in daily weather data from parquet
team_blob_url = blob_connect()
joined3M = spark.read.parquet(f"{team_blob_url}/ES/new_joins/3MO_schema")

# take only columns needed
# FIlter out all cancelled
df_clean = joined3M[['sched_depart_date_time_UTC','DEP_DELAY', 'ORIGIN', 'origin_3Hr_Precipitation', 'TAIL_NUM','OP_UNIQUE_CARRIER']].\
filter(joined3M['CANCELLED'] < 1).dropna()


# COMMAND ----------

joined3M.dtypes

# COMMAND ----------

df_clean.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform

# COMMAND ----------

# MAGIC %md
# MAGIC ### New feature: time in seconds since the beginning of Unix epoch. Only needed for window functions

# COMMAND ----------

# Make new column with time in seconds since the begining of Unix epoch
df_clean = df_clean.withColumn('time_long', df_clean.sched_depart_date_time_UTC.cast("long")).orderBy(df_clean.sched_depart_date_time_UTC)

# COMMAND ----------

# MAGIC %md
# MAGIC ### New feature: previos delay for this tail number  

# COMMAND ----------

hours = lambda i: i * 3600
Time_Tail_Window = Window.partitionBy('TAIL_NUM').orderBy(col('time_long')).rangeBetween(-hours(20), -hours(2))

df_clean = df_clean.withColumn("Prev_delay", last("DEP_DELAY").over(Time_Tail_Window)).fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### New feature: average delay for this airport number  

# COMMAND ----------

Time_Origin_Window = Window.partitionBy('ORIGIN').orderBy(col('time_long')).rangeBetween(-hours(6), -hours(2))

df_clean = df_clean.withColumn("Av_airport_delay", _mean("DEP_DELAY").over(Time_Origin_Window)).fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### New feature: sparse vector for the airline

# COMMAND ----------

categorical_indexer = StringIndexer(inputCol="OP_UNIQUE_CARRIER", outputCol="carrier_idx")
inputs = [categorical_indexer.getOutputCol()]
categorical_encoder = OneHotEncoder(inputCols=inputs, outputCols=["carrier_vec"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assemble all features in one vector

# COMMAND ----------

assembler = VectorAssembler().setInputCols(['carrier_vec', 'origin_3Hr_Precipitation', 'Prev_delay', "Av_airport_delay"] ).setOutputCol('feat_vec')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scale features to assist in regularization.

# COMMAND ----------

scaler = StandardScaler().setInputCol('feat_vec').setOutputCol('feat_scaled')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Make Labels column

# COMMAND ----------

# Make label column
df_clean = df_clean.withColumn('IsDelayed',  when(col("DEP_DELAY") >=15, 'Delayed').otherwise('On time'))
lbl_indexer = StringIndexer().setInputCol('IsDelayed').setOutputCol('label')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assemble and run transform pipeline

# COMMAND ----------

pipeline = Pipeline(stages=[categorical_indexer, categorical_encoder, assembler, scaler, lbl_indexer])
df_clean = pipeline.fit(df_clean).transform(df_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC #Train and Evaluate

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split

# COMMAND ----------

Rank_Window = Window.partitionBy().orderBy("sched_depart_date_time_UTC")
df_clean = df_clean.withColumn("rank", percent_rank().over(Rank_Window))
train_df = df_clean.where("rank <= .8").drop('rank', 'time_long','IsDelayed')
test_df = df_clean.where("rank > .8").drop('rank', 'time_long','IsDelayed')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Training

# COMMAND ----------

# Create an object model that is heavily biased toward LASSO regularization
lr = LR(featuresCol='feat_scaled', labelCol='label', maxIter=5)
model = lr.fit(train_df)
predictions = model.transform(test_df)
predictions.show()

# COMMAND ----------

# MAGIC %md
# MAGIC https://medium.com/swlh/logistic-regression-with-pyspark-60295d41221

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make predictions at various thresholds of what delay odds are considered a predicted delay.

# COMMAND ----------

# Convert probability output column to a column with probability of positive

def extract_prob(v):
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None

extract_prob_udf = udf(extract_prob, DoubleType())
predictions = predictions.withColumn("prob_pos", extract_prob_udf(col("probability")))

# COMMAND ----------

# Set decison cut offs
CutOffs = [0, 0.15, 0.16, 0.17, 0.19, 0.21, 0.25, 0.30, 0.80]

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
predictions = predictions.withColumns({'TP':make_TP(predictions.prob_pos, predictions.label), 'FP':make_FP(predictions.prob_pos, predictions.label)})

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
results_pd.to_csv('../Data/Trivial_LR_prec_rec.csv')
results_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## LR as a feature selector

# COMMAND ----------

# Create an object model that is heavily biased toward LASSO regularization
lr_selector = LR(featuresCol='feat_scaled', labelCol='label', maxIter=10, regParam=0.05, elasticNetParam=1)
model_selector = lr_selector.fit(train_df)
model_selector.coefficients

# COMMAND ----------

model_selector.summary.objectiveHistory

# COMMAND ----------


