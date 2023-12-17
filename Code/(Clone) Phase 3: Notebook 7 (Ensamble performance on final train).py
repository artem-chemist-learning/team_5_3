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
# MAGIC #### Helper functions

# COMMAND ----------

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

def evaluate_model(prediction, CutOffs, prob_pos, label):
    # Make arrays of the same length as the cut-off array
    # With 1 if the decision would be TP(FP) at this cut off
    prediction = prediction.withColumns({'TP':make_TP(col(prob_pos), col(label), CutOffs), 'FP':make_FP(col(prob_pos), col(label), CutOffs)})

    # Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
    num_cols = len(CutOffs)
    TP_FP_pd = prediction.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                            array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                            _sum(col("label")).alias("Positives")
                            ).toPandas()

    # Convert the result into the pd df of precisions and recalls for each cu-off
    results_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
    results_pd['Precision'] = 100*results_pd['TP']/(results_pd['TP'] + results_pd['FP'])
    results_pd['Recall']= 100*results_pd['TP']/TP_FP_pd.iloc[0,2]
    return results_pd

def impute_precision(x,y, x_to_impute):
    int_idx = 0
    for i in range(len(x)-1):
        if ((x[i] > x_to_impute) & (x[i+1] < x_to_impute)):
            int_idx = i
    impute_value = y[int_idx+1] - (y[int_idx+1] - y[int_idx])* ((x[int_idx+1]-x_to_impute)/(x[int_idx+1]-x[int_idx]))
    return impute_value

extract_prob_udf = udf(lambda x: float(x[1]) , DoubleType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read data from storage

# COMMAND ----------

# Load data
test_data = spark.read.parquet(f"{team_blob_url}/BK/final_test")

#Load models
#Trivial_LR_model = mlflow.spark.load_model('runs:/aec071ed330042549cfcfda11dc2d007/model')
Eng_LR_model = mlflow.spark.load_model('runs:/2f007c3381f64835a0512b2760755b1b/model')
#MLP_model = mlflow.spark.load_model('runs:/52bd40bf8dd84ed3973e7de4b0783208/model')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read RF preditions
# MAGIC

# COMMAND ----------

RFPred = spark.read.parquet(f"{team_blob_url}/ES/RF/Model4_finaltest_test")

# Convert probability output column to a column with probability of positive
RFPred = RFPred.withColumns({"rf_prob_pos": extract_prob_udf(col("probability"))})

CutOffs = [0, 0.4, 0.45, 0.47, 0.5, 0.6, 0.7, 0.8]

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

# Make arrays of the same length as the cut-off array
# With 1 if the decision would be TP(FP) at this cut off
RFPred = RFPred.withColumns({'TP':make_TP(col('rf_prob_pos'), col('label')), 'FP':make_FP(col('rf_prob_pos'), col('label'))})

# Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
num_cols = len(CutOffs)
TP_FP_pd = RFPred.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                        array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                        _sum(col("label")).alias("Positives")
                        ).toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
rf_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
rf_pd['Precision'] = 100*rf_pd['TP']/(rf_pd['TP'] + rf_pd['FP'])
rf_pd['Recall']= 100*rf_pd['TP']/TP_FP_pd.iloc[0,2]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict with Engineered LR

# COMMAND ----------

pred_eng = Eng_LR_model.transform(test_data)

# Convert probability output column to a column with probability of positive
extract_prob_udf = udf(lambda x: float(x[1]) , DoubleType())
threshold = 0.4
pred_eng = pred_eng.withColumns({"eng_lr_prob_pos": extract_prob_udf(col("probability")),
                                 "eng_lr_pred_lbl": when(col("eng_lr_prob_pos") >=threshold, 1).otherwise(0),
                                   'label':  when(col("DEP_DELAY") >=15, 1).otherwise(0)  })

LREngPred = pred_eng[['sched_depart_date_time_UTC', 'TAIL_NUM', 'DEP_DELAY', 'label', 'eng_lr_prob_pos', "eng_lr_pred_lbl"]]
write_parquet_to_blob(LREngPred, 'BK/LREngPred_test')

CutOffs = [0, 0.30, 0.35, 0.37, 0.4, 0.45, 0.8]

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

# Make arrays of the same length as the cut-off array
# With 1 if the decision would be TP(FP) at this cut off
LREngPred =LREngPred.withColumns({'TP':make_TP(col('eng_lr_prob_pos'), col('label'))
                                , 'FP':make_FP(col('eng_lr_prob_pos'), col('label'))})

# Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
num_cols = len(CutOffs)
TP_FP_pd = LREngPred.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                        array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                        _sum(col("label")).alias("Positives")
                        ).toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
lr_eng_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
lr_eng_pd['Precision'] = 100*lr_eng_pd['TP']/(lr_eng_pd['TP'] + lr_eng_pd['FP'])
lr_eng_pd['Recall']= 100*lr_eng_pd['TP']/TP_FP_pd.iloc[0,2]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read MLP predictions

# COMMAND ----------


MLPpred = spark.read.parquet(f"{team_blob_url}/LH/MLP/mlp_final_test")
# Convert probability output column to a column with probability of positive
MLPpred = MLPpred.withColumns({"mlp_prob_pos": extract_prob_udf(col("mlp_prob_pos"))})

CutOffs = [0.30, 0.35, 0.37, 0.4, 0.45, 0.6, 0.7, 0.8]

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

# Make arrays of the same length as the cut-off array
# With 1 if the decision would be TP(FP) at this cut off
MLPpred = MLPpred.withColumns({'TP':make_TP(col('mlp_prob_pos'), col('label'))
                               , 'FP':make_FP(col('mlp_prob_pos'), col('label'))})

# Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
num_cols = len(CutOffs)
TP_FP_pd = MLPpred.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                        array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                        _sum(col("label")).alias("Positives")
                        ).toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
mlp_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
mlp_pd['Precision'] = 100*mlp_pd['TP']/(mlp_pd['TP'] + mlp_pd['FP'])
mlp_pd['Recall']= 100*mlp_pd['TP']/TP_FP_pd.iloc[0,2]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict with Baseline

# COMMAND ----------

# Set decison cut offs
CutOffs = [0, 2, 3, 5, 10, 30, 60]

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

# Make arrays of the same length as the cut-off array
# With 1 if the decision would be TP(FP) at this cut off
test_data = test_data.withColumns({'TP' : make_TP(col('Av_carrier_delay'), col('DEP_DEL15'))
                                ,'FP' : make_FP(col('Av_carrier_delay'), col('DEP_DEL15'))
                                })

# Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
num_cols = len(CutOffs)
TP_FP_pd = test_data.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                        array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                        _sum(col("DEP_DEL15")).alias("Positives")
                        ).toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
av_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
av_pd['Precision'] = 100*av_pd['TP']/(av_pd['TP'] + av_pd['FP'])
av_pd['Recall']= 100*av_pd['TP']/TP_FP_pd.iloc[0,2]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict Random

# COMMAND ----------

test_data = test_data.withColumn('rnd_pred', rand(seed = 42) )
# Set decison cut offs
CutOffs = [0, 0.30, 0.60, 0.90, 0.99]

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

# Make arrays of the same length as the cut-off array
# With 1 if the decision would be TP(FP) at this cut off
test_data = test_data.withColumns({'TP':make_TP(col('rnd_pred'), col('DEP_DEL15'))
                                , 'FP':make_FP(col('rnd_pred'), col('DEP_DEL15'))
                                })

# Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
num_cols = len(CutOffs)
TP_FP_pd = val_data.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                        array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                        _sum(col("DEP_DEL15")).alias("Positives")
                        ).toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
rnd_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
rnd_pd['Precision'] = 100*rnd_pd['TP']/(rnd_pd['TP'] + rnd_pd['FP'])
rnd_pd['Recall']= 100*rnd_pd['TP']/TP_FP_pd.iloc[0,2]

# COMMAND ----------

# Generate graph
dfs = {"LR": lr_eng_pd
        ,"Baseline": av_pd
        ,"Random": rnd_pd
        ,"MLP": mlp_pd
        ,'RF': rf_pd
        }

colors = {"LR":'g'
        ,"Baseline":'r'
        ,"Random":'black'
        ,"MLP":'b'
        ,"RF":"magenta"
        }


for df in dfs.values():
  df.drop(df[df.Precision < 1].index, inplace=True)
  df.drop(df[df.Recall < 1].index, inplace=True)
  df.drop(df[df.Precision > 90].index, inplace=True)

# Instantiate figure and axis
num_rows = 1
num_columns = 1
fig, axes = plt.subplots(num_rows, num_columns, sharex=True)
fig.set_figheight(10)
fig.set_size_inches(8, 6)

#Fill the axis with data
for name, df in dfs.items():
  axes.plot(df.Recall, df.Precision, label = name, color = colors[name])
  axes.scatter(df.Recall, df.Precision, color =  colors[name])
  # Write cutoff vaulues on the graph
  for index in range(len(df.Cutoff)):
    axes.text(df.Recall[index]-0.02, 1 + df.Precision[index], df.Cutoff[index], size=7)

# Draw a vertical line to show 80% recall
axes.axvline(x=80, ymin=0.05, ymax=0.45, color='gray', ls = '--')
axes.text(75, 40, '80% Recall', size=12)

#Set legend position
axes.legend(loc = 'upper right')

#Setup the x and y 
axes.set_ylabel('Precision')
axes.set_xlabel('Recall')
axes.set_ylim(5, 80)

# Remove the bounding box to make the graphs look less cluttered
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)

plt.show()
fig.savefig(f"../Images/Models_on_test.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------

# Show the estimated final metric for each model
prec_dic = {}
for name, df in dfs.items():
    prec_dic[name] = [round(impute_precision(df['Recall'], df['Precision'], 80), 1)]

prec_df = pd.DataFrame.from_dict(prec_dic)
prec_df

# COMMAND ----------


