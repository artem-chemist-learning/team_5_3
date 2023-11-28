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

# Take only data needed
#take only columns needed
df_clean = joined3M[['sched_depart_date_time_UTC','DEP_DELAY', 'ORIGIN', 'origin_3Hr_Precipitation', 'TAIL_NUM','OP_UNIQUE_CARRIER']].dropna()

#df_clean = df_clean.withColumn('sched_depart_date_time_UTC', to_timestamp(df_clean['sched_depart_date_time_UTC']))
#df_clean = df_clean.withColumn('DEP_DELAY', df_clean['DEP_DELAY'].cast(DoubleType()))


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
# MAGIC ### New feature: sparse vector for the airline

# COMMAND ----------

categorical_indexer = StringIndexer(inputCol="OP_UNIQUE_CARRIER", outputCol="carrier_idx")
inputs = [categorical_indexer.getOutputCol()]
categorical_encoder = OneHotEncoder(inputCols=inputs, outputCols=["carrier_vec"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assemble all features in one vector

# COMMAND ----------

assembler = VectorAssembler().setInputCols(['carrier_vec', 'origin_3Hr_Precipitation', 'Prev_delay'] ).setOutputCol('feat_vec')

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

lr = LR(featuresCol='feat_scaled', labelCol='label', maxIter=5)
predictions = lr.fit(train_df).transform(test_df)
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

# Convert label into a bool type column for fater processing
predictions = predictions.withColumn("bool_lbl", predictions.label > 0)

# COMMAND ----------

#Get number of delated and total number of flights
pos_stats = predictions.select(
    _sum(col('label')).alias('num_delayed')
    ).collect()

Positive = pos_stats[0]['num_delayed']
Total = predictions.count()
print(f"Actually delayed: {Positive}, Total flights:{Total}")

# COMMAND ----------

cut_offs = [0, 0.15, 0.16, 0.17, 0.19, 0.21, 0.25, 0.30, 0.80]
data = 0.6
bool_lbl = True

#     return [ ( int((data >= cut_off)  &  bool_lbl  ),  int((data >= cut_off)  &  ~bool_lbl )   ) for cut_off in CutOffs ]

def TP(prob_pos, label):
    CutOffs =  [0, 0.15, 0.16, 0.17, 0.19, 0.21, 0.25, 0.30, 0.80]
    return [ 1 if (prob_pos >= cut_off) and (label > 0)  else 0 for cut_off in CutOffs]
def FP(prob_pos, label):
    CutOffs =  [0, 0.15, 0.16, 0.17, 0.19, 0.21, 0.25, 0.30, 0.80]
    return [ 1 if (prob_pos >= cut_off) and (label < 1)  else 0 for cut_off in CutOffs]

make_TP = udf(TP,  ArrayType(IntegerType()))
make_FP = udf(FP,  ArrayType(IntegerType()))

# COMMAND ----------

predictions = predictions.withColumns({'TP':make_TP(predictions.prob_pos, predictions.label), 'FP':make_FP(predictions.prob_pos, predictions.label)})

# COMMAND ----------

predictions.take(1)

# COMMAND ----------

# Make preditions, given cutoff
def prec_rec(cut_off, num_positive, data):
    df_pred_stats = data.select(
        _sum(  ((data.prob_pos >= cut_off)  &  data.bool_lbl  ).cast('integer')   ).alias('TP'),
        _sum(  ((data.prob_pos >= cut_off)  &  ~data.bool_lbl ).cast('integer')   ).alias('FP')
    ).collect()

    TP = df_pred_stats[0]['TP']
    FP = df_pred_stats[0]['FP'] 

    precision = 100*TP/(TP+FP)
    recall = 100*TP/num_positive

    return precision, recall
results = []
cut_offs = [0, 0.15, 0.16, 0.17, 0.19, 0.21, 0.25, 0.30, 0.60]
for i in cut_offs:
    results.append(prec_rec(i, Positive, predictions))

results_pd = pd.DataFrame(results)
results_pd.columns = ['Precision', 'Recall']
results_pd['Cutoff'] = cut_offs
results_pd.to_csv('../Data/Trivial_LR_prec_rec.csv')
results_pd
