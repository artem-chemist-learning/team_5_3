# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Logisitc regression w engineered features on the unbalanced dataset

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
from pyspark.ml.classification import LogisticRegression as LR

team_blob_url = blob_connect()

# COMMAND ----------

!pwd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read from storage

# COMMAND ----------

# read in daily weather data from parquet

joined = spark.read.parquet(f"{team_blob_url}/BK/clean_5yr_WITHOUT_2019_eng")


# COMMAND ----------

bad_days = ['04', '05']
bad_airlines = ['WN', 'DL', 'B6', 'AS', 'F9', 'HA']
bad_origins = ['ORD', 'DFW', 'DEN', 'LAX', 'SFO', 'LGA', 'LAS', 'HNL', 'EWR']
bad_destinations = ['ATL', 'SFO', 'IAH', 'EWR', 'LGA', 'JFK']
bad_months = ['4', '9', '10']
sig_num =['origin_DailySnowfall', 'origin_DailyPrecipitation', 'origin_DailyDepartureFromNormalAverageTemperature', 'origin_DailyAverageDryBulbTemperature',       
          'origin_DailyAverageRelativeHumidity', 'origin_DailySustainedWindSpeed',  
          'dest_DailySnowfall',  'dest_DailyDepartureFromNormalAverageTemperature', 'dest_DailyAverageDryBulbTemperature','dest_DailyAverageRelativeHumidity',  'dest_DailySustainedWindSpeed',
          'origin_HourlyDryBulbTemperature',  'origin_HourlyPrecipitation',  'origin_HourlyVisibility',
          'dest_HourlyDryBulbTemperature',  'dest_HourlyPrecipitation', 'dest_HourlyVisibility']

categorical_features = ['DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'MONTH']

eng_features = ['origin_3Hr_DryBulbTemperature', 'origin_3Hr_PressureChange', 'origin_3Hr_StationPressure',  'origin_3Hr_Visibility', 'dest_3Hr_DryBulbTemperature',  'dest_3Hr_PressureChange', 'dest_3Hr_StationPressure',  'dest_3Hr_Precipitation',  'dest_3Hr_Visibility',  "Av_airport_delay", "Prev_delay_tailnum", "Av_carrier_delay", "weekly_flights_tailnum", 'airport_congestion', 'hourly_flights_origin', 'precip_severity','snow_severity']

aux_features = ['sched_depart_date_time_UTC', 'TAIL_NUM', 'DEP_DELAY']

# COMMAND ----------

# take only columns needed
df_clean = joined[aux_features + categorical_features + sig_num + eng_features].dropna()

df_clean = df_clean.withColumns({'DAY_OF_WEEK': df_clean.DAY_OF_WEEK.cast('string')
                                 ,'MONTH': df_clean.MONTH.cast('string')
                                })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate new columns for categorical values.
# MAGIC ### These columns only have slected levels of the category 

# COMMAND ----------

def select_month(v):
    try:
        if v in bad_months:
            return v
        else:
            return 'good_month'
    except ValueError:
        return None
select_month_udf = udf(select_month, StringType())

def select_airlines(v):
    try:
        if v in bad_airlines:
            return v
        else:
            return 'good_carrier'
    except ValueError:
        return None
    
select_airlines_udf = udf(select_airlines, StringType())


def select_origins(v):
    try:
        if v in bad_origins:
            return v
        else:
            return 'good_origin'
    except ValueError:
        return None
select_origins_udf = udf(select_origins, StringType())

def select_destinations(v):
    try:
        if v in bad_destinations:
            return v
        else:
            return 'good_destination'
    except ValueError:
        return None
select_destinations_udf = udf(select_destinations, StringType())

def select_day(v):
    try:
        if v in bad_days:
            return v
        else:
            return 'good_day'
    except ValueError:
        return None
select_day_udf = udf(select_day, StringType())

df_clean = df_clean.withColumns({
                                'MONTH':select_month_udf(col('MONTH'))
                                ,'ORIGIN':select_origins_udf(col('ORIGIN'))
                                ,'OP_UNIQUE_CARRIER':select_airlines_udf(col('OP_UNIQUE_CARRIER'))
                                ,'DAY_OF_WEEK':select_day_udf(col('DAY_OF_WEEK'))
                                ,'DEST':select_destinations_udf(col('DEST'))
                                 })
                                 

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sparse vectors for categorical features

# COMMAND ----------

indexers = [StringIndexer(inputCol=f"{Cat_Col}", outputCol=f"{Cat_Col}_idx") for Cat_Col in categorical_features]

encoder = OneHotEncoder(
    inputCols=[indexer.getOutputCol() for indexer in indexers],
    outputCols=["{0}_vec".format(indexer.getOutputCol()) for indexer in indexers]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assemble all features in one vector

# COMMAND ----------

numerical_features = sig_num + eng_features
input_cols = encoder.getOutputCols() + numerical_features
assembler = VectorAssembler().setInputCols(input_cols).setOutputCol('feat_vec')

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

pipeline = Pipeline(stages=indexers+[encoder, assembler, scaler, lbl_indexer])
df_clean = pipeline.fit(df_clean).transform(df_clean)

# Generate a human-readable list of features
# Read label maps for categorical features
lbl_map = {c.name: c.metadata["ml_attr"]["vals"] for c in df_clean.schema.fields if c.name.endswith("_idx")}

# Assemble formatted list of these features
lbl_lst = []
lbl_base_dic = {}
lbl_idx = 0
for feature in categorical_features:
    for key, value in lbl_map.items():
        if feature == key[:-4]:
            for level in value[:-1]:
                lbl_lst.append(f'{lbl_idx}: {feature} = {level}')
                lbl_idx +=1
            lbl_base_dic[feature] = value[-1]

# Add numerical features to this list features
for ft in numerical_features:
    lbl_lst.append(f'{lbl_idx}: {ft}')
    lbl_idx +=1

# COMMAND ----------

# MAGIC %md
# MAGIC #Train and Evaluate

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split

# COMMAND ----------

number_of_folds = 5
test_train_sets = create_validation_blocks(df_clean, "sched_depart_date_time_UTC", blocks=number_of_folds, split=0.8)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Training

# COMMAND ----------

# Create an object model that is heavily biased toward LASSO regularization
lrs = []
models = []
predictions = []
predictions_train = []
for train_df, test_df in test_train_sets:
    lrs.append(LR(featuresCol='feat_scaled', labelCol='label', maxIter=5, regParam=0.005, elasticNetParam=1))
    models.append(lrs[-1].fit(train_df))
    predictions.append(models[-1].transform(test_df))
    predictions_train.append(models[-1].transform(train_df))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Make predictions at various thresholds of what delay odds are considered a predicted delay.

# COMMAND ----------

# Combine predictions made on test data from all blocks into a single dataframe
prediction = predictions[0]
for p in predictions[1:]:
    prediction = prediction.union(p)

# Combine predictions made on train data from all blocks into a single dataframe
prediction_train = predictions_train[0]
for p in predictions_train[1:]:
    prediction_train = prediction_train.union(p)

# Convert probability output column to a column with probability of positive
def extract_prob(v):
    try:
        return float(v[1])
    except ValueError:
        return None

extract_prob_udf = udf(extract_prob, DoubleType())
prediction = prediction.withColumn("prob_pos", extract_prob_udf(col("probability")))
prediction_train = prediction_train.withColumn("prob_pos", extract_prob_udf(col("probability")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perfomance on test data

# COMMAND ----------

# Set decison cut offs
CutOffs = [0, 0.12, 0.15, 0.20, 0.25, 0.40, 0.60, 0.80]

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
prediction = prediction.withColumns({'TP':make_TP(prediction.prob_pos, prediction.label), 'FP':make_FP(prediction.prob_pos, prediction.label)})

# Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
num_cols = len(CutOffs)
TP_FP_pd = prediction.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                        array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                        _sum(col("label")).alias("Positives")
                        )\
                        .toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
results_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
results_pd['Precision'] = 100*results_pd['TP']/(results_pd['TP'] + results_pd['FP'])
results_pd['Recall']= 100*results_pd['TP']/TP_FP_pd.iloc[0,2]
results_pd.to_csv('../Data/Eng_LR_validation_unbalanced.csv')
results_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance on train data

# COMMAND ----------

# Generate these arrays in the dataframe returned by prediction
prediction_train = prediction_train.withColumns({
    'TP':make_TP(prediction_train.prob_pos, prediction_train.label)
    , 'FP':make_FP(prediction_train.prob_pos, prediction_train.label)})

# Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
num_cols = len(CutOffs)
TP_FP_pd = prediction_train.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                        array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                        _sum(col("label")).alias("Positives")
                        )\
                        .toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
results_pd_train= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
results_pd_train['Precision'] = 100*results_pd_train['TP']/(results_pd_train['TP'] + results_pd_train['FP'])
results_pd_train['Recall']= 100*results_pd_train['TP']/TP_FP_pd.iloc[0,2]
results_pd_train.to_csv('../Data/Eng_LR_train_unbalanced.csv')
results_pd_train

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Visualize overfitting

# COMMAND ----------

def impute_precision(x,y, x_to_impute):
    int_idx = 0
    for i in range(len(x)-1):
        if ((x[i] > x_to_impute) & (x[i+1] < x_to_impute)):
            int_idx = i
    impute_value = y[int_idx+1] - (y[int_idx+1] - y[int_idx])* ((x[int_idx+1]-x_to_impute)/(x[int_idx+1]-x[int_idx]))
    return impute_value

# COMMAND ----------

dfs = [results_pd, results_pd_train]

for df in dfs:
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
axes.plot(results_pd.Recall, results_pd.Precision, label = "Validation", color = 'g')
axes.scatter(results_pd.Recall, results_pd.Precision, color = 'g')

axes.plot(results_pd_train.Recall, results_pd_train.Precision, label = "Train", color = 'r') 
axes.scatter (results_pd_train.Recall, results_pd_train.Precision, color = 'r') 

# Draw a vertical line to show 80% recall
axes.axvline(x=80, ymin=0.05, ymax=0.45, color='gray', ls = '--')
axes.text(70, 40, '80% Recall', size=12)

# Write cutoff vaulues on the graph
for index in range(len(results_pd.Cutoff)):
  axes.text(results_pd.Recall[index]-0.02, 1 + results_pd.Precision[index], results_pd.Cutoff[index], size=9)

for index in range(len(results_pd_train.Cutoff)):
  axes.text(results_pd_train.Recall[index]-0.02, 1 + results_pd_train.Precision[index], results_pd_train.Cutoff[index], size=9)

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
fig.savefig(f"../Images/Eng_LR_overfitting_unbalanced.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------

df_names = ['Validation', 'Training']
prec_dic = {}
for df, name in zip(dfs, df_names):
    prec_dic[name] = [round(impute_precision(df['Recall'], df['Precision'], 80), 1)]

prec_df = pd.DataFrame.from_dict(prec_dic)
prec_df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Get the human readable names for the significant features

# COMMAND ----------

common_features = {}
for i in range(number_of_folds):
    Sig_w_dic = {}
    for lbl, weight in zip(lbl_lst, list(models[i].coefficients)):
        if weight*weight >0.0001:
            Sig_w_dic[lbl] = weight
            if lbl in common_features.keys():
                common_features[lbl] +=1
            else: common_features[lbl] = 1
    print ('-'*5)
    print(f"{i}'th Weights")
    for key, value in Sig_w_dic.items():
        print(f'{key}\t: {value}')

print ('_'*30)
print('Base categoties')
for key, value in lbl_base_dic.items():
    print(f'{key} : {value}')

print ('_'*30)
print('Significant features:')
for ft, frq in common_features.items():
    if frq >=4:
        print(ft)

# COMMAND ----------


