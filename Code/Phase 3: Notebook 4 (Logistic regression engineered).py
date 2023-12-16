# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Logistic regression with engeneered features and balancing of training set

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

# MAGIC %md
# MAGIC ## Read from storage

# COMMAND ----------

joined = spark.read.parquet(f"{team_blob_url}/BK/pure_train")

# COMMAND ----------

# Read features selected by trivial LR
with open('lr_trivial_sig_feat.json', 'r') as fp:
    features = json.load(fp)

features = dict(filter(lambda x: True if x[1] >=2 else False, features.items()))

all_simple_features = ['DISTANCE', 'origin_DailySnowfall',  'origin_DailyPrecipitation'  , 'origin_DailyDepartureFromNormalAverageTemperature' ,  'origin_DailyAverageDryBulbTemperature',  'origin_DailyAverageRelativeHumidity' ,  'origin_DailyAverageStationPressure' , 'origin_DailySustainedWindDirection' ,  'origin_DailySustainedWindSpeed' ,  'dest_DailySnowfall' , 'dest_DailyPrecipitation',   'dest_DailyDepartureFromNormalAverageTemperature',  'dest_DailyAverageDryBulbTemperature',   'dest_DailyAverageRelativeHumidity',  'dest_DailyAverageStationPressure' ,  'dest_DailySustainedWindDirection',  'dest_DailySustainedWindSpeed' ,  'origin_HourlyDryBulbTemperature' ,   'origin_HourlyWindDirection' , 'origin_HourlyPrecipitation',  'origin_HourlyVisibility', 'dest_HourlyDryBulbTemperature', 'dest_HourlyStationPressure' , 'dest_HourlyWindDirection' , 'dest_HourlyPrecipitation' , 'dest_HourlyVisibility' ]

all_categorical_features = ['DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'MONTH', 'ORIGIN', 'DEST', 'is_holiday_window']

aux_features = ['sched_depart_date_time_UTC', 'DEP_DELAY', 'TAIL_NUM', 'DEP_DEL15']

bad_days = []
bad_airlines = []
bad_origins = []
bad_destinations = []
bad_months = []

bad_lsts = [bad_days, bad_airlines, bad_months, bad_origins, bad_destinations]

sig_num = []
categorical_features = []

# Fill the list of the simple features that came out to be significant in the trivial case
for f in all_simple_features:
    for key in features.keys():
        if f in key:
           sig_num.append(key.split(' ')[-1])

# Fill the lists of the siginificance categories of categorical features
for f, f_lst in zip(all_categorical_features, bad_lsts):
    for key in features.keys():
        if f in key:
            f_lst.append(key.split(' ')[-1])

# Fill the list of the categorical features that came out to be significant in the trivial case
for f, f_lst in zip (all_categorical_features , bad_lsts):
    if len(f_lst) > 0:
        categorical_features.append(f)  

# COMMAND ----------

eng_features = ['origin_3Hr_DryBulbTemperature', 'origin_3Hr_PressureChange', 'origin_3Hr_StationPressure',  'origin_3Hr_Visibility', 'dest_3Hr_DryBulbTemperature',  'dest_3Hr_PressureChange', 'dest_3Hr_StationPressure',  'dest_3Hr_Precipitation',  'dest_3Hr_Visibility',  "Av_airport_delay", "Prev_delay_tailnum", "Av_carrier_delay", "weekly_flights_tailnum", 'airport_congestion', 'hourly_flights_origin', 'precip_severity','snow_severity']

# COMMAND ----------

# take only columns needed
df_clean = joined[aux_features + categorical_features + sig_num + eng_features].dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate new columns for categorical values.
# MAGIC ### These columns only have selected levels of the category 

# COMMAND ----------

select_month_udf = udf(lambda x: x if (str(x) in bad_months) else 'good_month', StringType())
select_airlines_udf = udf(lambda x: x if( x in bad_airlines) else 'good_carrier', StringType())
select_origins_udf = udf(lambda x: x if (x in bad_origins) else 'good_origin', StringType())
select_destinations_udf = udf(lambda x: x if( x in bad_destinations) else 'good_destination', StringType())
select_day_udf = udf(lambda x: x if (x in bad_days) else 'good_day', StringType())

df_clean = df_clean.withColumns({
                                'MONTH':select_month_udf(col('MONTH'))
                                ,'ORIGIN':select_origins_udf(col('ORIGIN'))
                                ,'OP_UNIQUE_CARRIER':select_airlines_udf(col('OP_UNIQUE_CARRIER'))
                                ,'DAY_OF_WEEK':select_day_udf(col('DAY_OF_WEEK'))
                                ,'DEST':select_destinations_udf(col('DEST'))
                                 })
                                 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split

# COMMAND ----------

number_of_folds = 3
test_val_sets = create_validation_blocks(df_clean, "sched_depart_date_time_UTC", blocks=number_of_folds, split=0.8)
balanced_sets = []

# Fix the imbalance on the train_df by dropping ~80% of random on-time flights
on_time_correction = 0.2 # We assume thwere is about 5 on-time flaight for each delayed
for train_df, val_df in test_val_sets:
    # Set an aux "to keep" column that is true is the flight is delayed, or 20% chance true if the flight is on time
    train_df = train_df.withColumn('to_keep', when(
        ((rand(seed = 42) < on_time_correction) & (col('DEP_DEL15') < 1)) | (col('DEP_DEL15') > 0)
        , True   ))
    # Take only records that are flagged to keep in the previous step
    train_df = train_df.filter(train_df.to_keep).drop('to_keep')

    balanced_sets.append((train_df, val_df))

# COMMAND ----------

# MAGIC %md
# MAGIC # Assemble stages for transform pipeline

# COMMAND ----------

# Sparse vectors for categorical features
indexers = [StringIndexer(inputCol=f"{Cat_Col}", outputCol=f"{Cat_Col}_idx", stringOrderType = 'alphabetAsc') for Cat_Col in categorical_features]

encoder = OneHotEncoder(
    inputCols=[indexer.getOutputCol() for indexer in indexers],
    outputCols=["{0}_vec".format(indexer.getOutputCol()) for indexer in indexers]
)

# Assemble all features in one vector
numerical_features = sig_num + eng_features
input_cols = encoder.getOutputCols() + numerical_features
assembler = VectorAssembler().setInputCols(input_cols).setOutputCol('feat_vec')

# Scale features to assist in regularization.
scaler = StandardScaler().setInputCol('feat_vec').setOutputCol('feat_scaled')

# Make Labels column
lbl_indexer = StringIndexer(stringOrderType = 'alphabetAsc').setInputCol('DEP_DEL15').setOutputCol('label')

# COMMAND ----------

# MAGIC %md
# MAGIC #Train and Evaluate

# COMMAND ----------

# MAGIC %md
# MAGIC ##Training

# COMMAND ----------

def make_feature_map(df):
    # Generate a human-readable list of features
    # Read label maps for categorical features
    lbl_map = {c.name: c.metadata["ml_attr"]["vals"] for c in df.schema.fields if c.name.endswith("_idx")}

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
    return (lbl_lst, lbl_base_dic)

# COMMAND ----------

# Create an object model that is heavily biased toward LASSO regularization
pipelines = []
predictions = []
predictions_train = []
feature_maps = []
for train_df, val_df in balanced_sets:
    lr = LR(featuresCol='feat_scaled', labelCol='label', maxIter=5, regParam=0.005, elasticNetParam=1)
    pipe = Pipeline(stages=indexers+[encoder, assembler, scaler, lbl_indexer, lr])
    pipelines.append(pipe.fit(train_df))
    predictions.append(pipelines[-1].transform(val_df))
    predictions_train.append(pipelines[-1].transform(train_df))
    feature_maps.append(make_feature_map(train_df))

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
extract_prob_udf = udf(lambda x: float(x[1]), DoubleType())
prediction = prediction.withColumn("prob_pos", extract_prob_udf(col("probability")))
prediction_train = prediction_train.withColumn("prob_pos", extract_prob_udf(col("probability")))

# COMMAND ----------

# MAGIC %md
# MAGIC # Performance evaluation
# MAGIC ### Perfomance on test data

# COMMAND ----------

# Set decison cut offs
CutOffs = [0, 0.30, 0.40, 0.45, 0.60, 0.80]

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
prediction = prediction.withColumns({'TP':make_TP(prediction.prob_pos, prediction.DEP_DEL15), 'FP':make_FP(prediction.prob_pos, prediction.DEP_DEL15)})

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
results_pd.to_csv('../Data/Eng_LR_validation.csv')
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
                        ).toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
results_pd_train= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
results_pd_train['Precision'] = 100*results_pd_train['TP']/(results_pd_train['TP'] + results_pd_train['FP'])
results_pd_train['Recall']= 100*results_pd_train['TP']/TP_FP_pd.iloc[0,2]
results_pd_train.to_csv('../Data/Eng_LR_train.csv')
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

dfs = {"Validation" :results_pd, "Train":results_pd_train}
colors = {"Validation" :'g', "Train":'r'}

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
axes.text(70, 40, '80% Recall', size=12)

#Set legend position
axes.legend(loc = 'upper right')

#Setup the x and y 
axes.set_ylabel('Precision')
axes.set_xlabel('Recall')
#axes.set_ylim(5, 80)

# Remove the bounding box to make the graphs look less cluttered
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)

plt.show()
fig.savefig(f"../Images/Eng_LR_overfitting.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------

prec_dic = {}
for name, df in dfs.items():
    prec_dic[name] = [round(impute_precision(df['Recall'], df['Precision'], 80), 1)]

prec_df = pd.DataFrame.from_dict(prec_dic)
prec_df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Get the human readbale names for the significant features

# COMMAND ----------

common_features = {}
model_num = 0
for lbl_lst, lbl_base_dic in feature_maps:
    Sig_w_dic = {}
    for lbl, weight in zip(lbl_lst, list(pipelines[model_num].stages[-1].coefficients)):
        if weight*weight >0.0001:
            Sig_w_dic[lbl] = weight
            ft_name = lbl.split(' ')[-1]
            if ft_name in common_features.keys():
                common_features[ft_name] +=1
            else: common_features[ft_name] = 1
    print ('-'*5)
    print(f"{model_num}'th Weights")
    for key, value in Sig_w_dic.items():
        print(f'{key}\t: {value}')
    print ('-'*5)
    print(f"{model_num}'th Base categoties")
    for key, value in lbl_base_dic.items():
        print(f'{key} : {value}')
    model_num+=1
print ('_'*30)

common_features = dict(filter(lambda x: True if x[1] >=2 else False, common_features.items()))

print ('_'*30)
print('Significant features:')
for ft, frq in common_features.items():
    print(ft)

# Save common features as a csv
with open('lr_eng_sig_feat.json', 'w') as fp:
    json.dump(common_features, fp)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrain on the selected set of features

# COMMAND ----------

# Read features selected by engineered LR
with open('lr_eng_sig_feat.json', 'r') as fp:
    features = json.load(fp)
 
features = dict(filter(lambda x: True if x[1] >=2 else False, features.items()))

bad_days = []
bad_airlines = []
bad_origins = []
bad_destinations = []
bad_months = []

bad_lsts = [bad_days, bad_airlines, bad_months, bad_origins, bad_destinations]

sig_num = []
sig_categorical_features = []

# Fill the list of the simple features that came out to be significant in the trivial case
for f in all_simple_features + eng_features:
    for key in features.keys():
        if f in key:
           sig_num.append(key.split(' ')[-1])

# Fill the lists of the siginificance categories of categorical features
for f, f_lst in zip(categorical_features, bad_lsts):
    for key in features.keys():
        if f in key:
            f_lst.append(key.split(' ')[-1])

# Fill the list of the categorical features that came out to be significant in the trivial case
for f, f_lst in zip (categorical_features , bad_lsts):
    if len(f_lst) > 0:
        sig_categorical_features.append(f)  

# COMMAND ----------

sig_num

# COMMAND ----------

test_train_sets = create_validation_blocks(joined[aux_features + sig_num], "sched_depart_date_time_UTC", blocks=1, split=0.01)

# Fix the imbalance on the train_df by dropping ~80% of random on-time flights
on_time_correction = 0.2 # We assume thwere is about 5 on-time flaight for each delayed
train_new = test_train_sets[0][1]

# Set an aux "to keep" column that is true is the flight is delayed, or 20% chance true if the flight is on time
train_new = train_new.withColumn('to_keep', when(
    ((rand(seed = 42) < on_time_correction) & (col('DEP_DEL15') < 1)) | (col('DEP_DEL15') > 0)
    , True   ))
# Take only records that are flagged to keep in the previous step
train_new = train_new.filter(train_new.to_keep).drop('to_keep')

# COMMAND ----------

# Assemble all features in one vector
input_cols = sig_num
assembler_new = VectorAssembler().setInputCols(input_cols).setOutputCol('feat_vec')

# Scale features to assist in regularization.
scaler_new = StandardScaler().setInputCol('feat_vec').setOutputCol('feat_scaled')

# Make Labels column
lbl_indexer_new = StringIndexer(stringOrderType = 'alphabetAsc').setInputCol('DEP_DEL15').setOutputCol('label')

LR_new = LR(featuresCol='feat_scaled', labelCol='label', maxIter=5, regParam=0.005, elasticNetParam=1)

pipeline_new = Pipeline(stages=[assembler_new, scaler_new, lbl_indexer_new, LR_new])
pipeline_new_model = pipeline_new.fit(train_new)
