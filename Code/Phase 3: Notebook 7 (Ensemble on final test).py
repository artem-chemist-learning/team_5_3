# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Predictions using Logistic regression model

# COMMAND ----------

# importing custom functions
from Code.funcs import blob_connect, write_parquet_to_blob, create_validation_blocks
import csv
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, FloatType, DoubleType, ArrayType, StringType
from pyspark.sql.functions import size, to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when, collect_list, udf, date_trunc, count, lag, first, last, percent_rank, array
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, IndexToString, StringIndexerModel
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression as LR, LogisticRegressionModel as LR_model
from pyspark.mllib.tree import RandomForest, RandomForestModel

import mlflow

team_blob_url = blob_connect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read data from storage

# COMMAND ----------

# Load data
# Assumes the following columns: ['sched_depart_date_time_UTC', 'TAIL_NUM', 'label', 'xxx', 'yyy']
# xxx: predicted probabilities; yyy: predicted labels. Both columns should have df-specific names
LREngPred = spark.read.parquet(f"{team_blob_url}/BK/LREngPred_test")
MLPpred = spark.read.parquet(f"{team_blob_url}/LH/MLP/mlp_final_test")
RFPred = spark.read.parquet(f"{team_blob_url}/ES/RF/Model4_finaltest_test")

# COMMAND ----------

# Convert probability output column to a column with probability of positive
extract_prob_udf = udf(lambda x: float(x[1]) , DoubleType())
RFPred = RFPred.withColumns({"rf_prob_pos": extract_prob_udf(col("probability"))})
MLPpred = MLPpred.withColumns({"mlp_prob_pos": extract_prob_udf(col("mlp_prob_pos"))})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assemble all predictions into one dataframe

# COMMAND ----------

df1_a = MLPpred.alias("df1_a")
df2_a = LREngPred.alias("df2_a")
df3_a = RFPred.alias("df3_a")


df = df1_a.join(df2_a, (col('df1_a.sched_depart_date_time_UTC') == col('df2_a.sched_depart_date_time_UTC')) &  (col('df1_a.TAIL_NUM') == col('df2_a.TAIL_NUM'))   ) \
    .select(df1_a['sched_depart_date_time_UTC'], df1_a['TAIL_NUM'], df2_a['DEP_DELAY'], df1_a['label'],
            df1_a['mlp_prob_pos'], df1_a["mlp_pred_lbl"],
            df2_a['eng_lr_prob_pos'], df2_a["eng_lr_pred_lbl"])


# COMMAND ----------

df_a = df.alias("df_a")
df3_a = RFPred.alias("df3_a")


df = df_a.join(df3_a, (col('df_a.sched_depart_date_time_UTC') == col('df3_a.sched_depart_date_time_UTC')) &  (col('df_a.TAIL_NUM') == col('df3_a.TAIL_NUM'))   ) \
    .select(df_a['sched_depart_date_time_UTC'], df_a['TAIL_NUM'], df_a['DEP_DELAY'], df_a['label'],
            df_a['mlp_prob_pos'], df_a["mlp_pred_lbl"],
            df_a['eng_lr_prob_pos'], df_a["eng_lr_pred_lbl"],
            df3_a['rf_prob_pos'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: Adjust predicted labels with a different probability cut_offs

# COMMAND ----------

cutoffs = [0.4, 0.4, 0.46]
lbls = ['eng_lr_pred_lbl', 'mlp_pred_lbl', 'rf_prob_lbl']
probs = ['eng_lr_prob_pos', 'mlp_prob_pos', 'rf_prob_pos']
df = df.withColumns({lbl:when(col(prob) >cutoff, 1).otherwise(0) for lbl, prob, cutoff in zip(lbls, probs, cutoffs) })

# COMMAND ----------

# MAGIC %md
# MAGIC ### Make predictions given weights

# COMMAND ----------

weights = [.1, .2, .7]
pred_lbls = ['mlp_prob_pos', 'eng_lr_prob_pos', 'rf_prob_pos']
df = df.withColumns({"score": sum([w*col(p) for w, p in zip(weights, pred_lbls)])
                     , "final":  when(col("score") >0.440, 1).otherwise(0)
                    })

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate Precision and recall

# COMMAND ----------

# Define functions to labeling a prediction as FP(TP) 
# Based on teh cut off
def TP(final, label):
    if (final >0) and (label > 0):
        return 1
    else:
        return 0
def FP(final, label):
    if (final > 0) and (label < 1):
        return 1
    else:
        return 0

# Define udfs based on these functions
# These udfs return arrays of the same length as the cut-off array
# With 1 if the decision would be TP(FP) at this cut off
make_TP = udf(TP,  IntegerType())
make_FP = udf(FP,  IntegerType())

# Generate TP and FP labels for each record
tp_fp_df = df.withColumns({'TP':make_TP(df.final, df.label), 'FP':make_FP(df.final, df.label)})

tp_fp_collected = tp_fp_df[['TP', 'FP', 'label']].groupBy().sum().collect()
TP = tp_fp_collected[0][0]
FP = tp_fp_collected[0][1]
Positives = tp_fp_collected[0][2]
Precision = 100*TP/(TP+FP)
Recall = 100*TP/Positives
F1 = 2*Precision*Recall/(Recall+Precision)
F2 = 5*Precision*Recall/(Recall+ 4*Precision)
print(f"Precision: {round(Precision,1)}, Recall: {round(Recall,1)}, F1: {round(F1,1)}, F2: {round(F2,1)}")
