# Databricks notebook source
import matplotlib.pyplot as plt
import seaborn as sns
from Code.funcs import blob_connect
import pandas as pd
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
import plotly.express as px
import mlflow
team_blob_url = blob_connect()

# COMMAND ----------

df_impute = spark.read.parquet(f"{team_blob_url}/BK/imputation_tracker")

# COMMAND ----------

#df_impute.display()
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show wider cell content
df_impute.iloc[:57]

# COMMAND ----------

df = df_dimension.toPandas()

# Moving the last row to the top (as shown before)
last_row = df.iloc[[-1]]
df = pd.concat([last_row, df.drop(df.tail(1).index)])

# Moving the 6th row to the 3rd position (as shown before)
sixth_row = df.iloc[[5]].copy()
df = pd.concat([df.iloc[:2], sixth_row, df.iloc[2:5], df.iloc[6:]])

# Moving the 7th row to the 6th position (as shown before)
seventh_row = df.iloc[[6]].copy()
df = pd.concat([df.iloc[:5], seventh_row, df.iloc[5:6], df.iloc[7:]])

# Switching the 2nd and 3rd rows
second_row = df.iloc[[1]].copy()
third_row = df.iloc[[2]].copy()

df.iloc[1] = third_row.values[0]
df.iloc[2] = second_row.values[0]
df

# COMMAND ----------

# MAGIC %md
# MAGIC # Decision Trees and Random Forest

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
from Code.funcs import blob_connect
import pandas as pd
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
import plotly.express as px
import mlflow
team_blob_url = blob_connect()
df = spark.read.parquet(f"{team_blob_url}/LH/1yr_clean_temp_2")
df = df.dropna()

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Collecting count data for categorical variables
count_day_of_week = df.groupBy('DAY_OF_WEEK').count().toPandas()
count_unique_carrier = df.groupBy('OP_UNIQUE_CARRIER').count().toPandas()
count_origin = df.groupBy('ORIGIN').count().toPandas()

# Renaming count column to indicate the count for respective variable
count_day_of_week.rename(columns={'count': 'Count_DAY_OF_WEEK'}, inplace=True)
count_unique_carrier.rename(columns={'count': 'Count_OP_UNIQUE_CARRIER'}, inplace=True)
count_origin.rename(columns={'count': 'Count_ORIGIN'}, inplace=True)

# Merging count data into a single DataFrame based on the categorical variables
merged_counts = pd.merge(count_day_of_week, count_unique_carrier, on='OP_UNIQUE_CARRIER')
merged_counts = pd.merge(merged_counts, count_origin, on='ORIGIN')

# Creating a pair plot for count data of categorical variables
sns.pairplot(merged_counts)
plt.show()

# COMMAND ----------

# Selecting non-delay-related columns of interest
interesting_columns = ['CRS_ELAPSED_TIME', 'DISTANCE', 'TAXI_OUT']

# Selecting and visualizing the specified columns
selected_interesting_df = df.select(interesting_columns).toPandas()

# Pairplot for the selected features
sns.pairplot(selected_interesting_df)
plt.show()


# COMMAND ----------


# Categorical variables: DAY_OF_WEEK, OP_UNIQUE_CARRIER, ORIGIN, DEST

# Count plot for DAY_OF_WEEK
plt.figure(figsize=(8, 6))
sns.countplot(data=df.toPandas(), x='DAY_OF_WEEK')
plt.title('Count of Flights by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()

# Count plot for OP_UNIQUE_CARRIER
plt.figure(figsize=(12, 6))
sns.countplot(data=df.toPandas(), x='OP_UNIQUE_CARRIER')
plt.title('Count of Flights by Carrier')
plt.xlabel('Carrier')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-labels for readability
plt.show()

# Count plot for ORIGIN (or DEST)
plt.figure(figsize=(14, 6))
sns.countplot(data=df.toPandas(), x='ORIGIN')
plt.title('Count of Flights by Origin Airport')
plt.xlabel('Origin Airport')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate x-labels for readability
plt.show()


# COMMAND ----------

# Assuming the columns: origin_airport_lat, origin_airport_lon, dest_airport_lat, dest_airport_lon

# Scatter plot for Origin Airport
plt.figure(figsize=(10, 8))
plt.scatter(df.select('origin_airport_lon').collect(), df.select('origin_airport_lat').collect(), alpha=0.5)
plt.title('Geographic Distribution of Origin Airports')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Scatter plot for Destination Airport
plt.figure(figsize=(10, 8))
plt.scatter(df.select('dest_airport_lon').collect(), df.select('dest_airport_lat').collect(), alpha=0.5)
plt.title('Geographic Distribution of Destination Airports')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()


# COMMAND ----------

# Bar plot for OP_UNIQUE_CARRIER (airlines)
plt.figure(figsize=(12, 6))
sns.countplot(data=df.toPandas(), x='OP_UNIQUE_CARRIER')
plt.title('Count of Flights by Airline')
plt.xlabel('Airline')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-labels for readability
plt.show()

# Bar plot for OP_CARRIER_AIRLINE_ID (airline IDs)
plt.figure(figsize=(14, 6))
sns.countplot(data=df.toPandas(), x='OP_CARRIER_AIRLINE_ID')
plt.title('Count of Flights by Airline ID')
plt.xlabel('Airline ID')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate x-labels for readability
plt.show()


# COMMAND ----------

import funcs

# reviewing the temperature based features. 
delay_columns = [
    'WEATHER_DELAY',
    'NAS_DELAY',
    'SECURITY_DELAY',
    'LATE_AIRCRAFT_DELAY'
]

funcs.pairplot(df, delay_columns)

# COMMAND ----------

# DBTITLE 1,Imports
# analysis requirements
from Code.funcs import blob_connect
import pandas as pd
import numpy as np
from pyspark.sql.functions import udf, isnan, when, count, col, expr, regexp_extract, size, to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when, collect_list, udf, date_trunc, count, lag, first, last, percent_rank, array
from pyspark.sql.types import StringType, IntegerType, FloatType, BooleanType
import pyspark.sql.functions as F
from pyspark.sql import types
import re
from pyspark.sql.functions import date_format
import matplotlib.pyplot as plt

# spark
from pyspark import SparkContext
from pyspark.sql import SparkSession

# cross val 
import statsmodels.api as sm
from pyspark.sql.window import Window

# log regression, decision tree, random forest
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler, IndexToString
from sklearn.tree import DecisionTreeRegressor

# evaluation 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, confusion_matrix, classification_report
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# COMMAND ----------

# DBTITLE 1,Load Dataset
# read in daily weather data from parquet
team_blob_url = blob_connect()
joined3M = spark.read.parquet(f"{team_blob_url}/ES/new_joins/1YR_schema").cache()

# COMMAND ----------

# DBTITLE 1,Specify, breakout all features
# List of string columns that need transforming
string_columns = [
    'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 
    'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_NM',
    'DEST_AIRPORT_ID', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 
    'DEST_STATE_NM', 'CANCELLATION_CODE', 'origin_airport_name', 
    'origin_station_name', 'origin_iata_code', 'origin_icao', 'origin_type', 
    'origin_region', 'dest_airport_name', 'dest_station_name', 
    'dest_iata_code', 'dest_icao', 'dest_type', 'dest_region'
]

date_columns = [
    'FL_DATE', 'sched_depart_date_time_UTC', 'four_hours_prior_depart_UTC',
    'two_hours_prior_depart_UTC', 'three_hours_prior_depart_UTC', 'origin_DATE',
    'origin_UTC', 'dest_DATE', 'dest_UTC'
]

#all other are numeric
numeric_columns = list(set(joined3M.columns) - set(string_columns) - set(date_columns))

# weird stuff happening
columns_to_remove = ['DEP_TIME_BLK', 'ARR_TIME_BLK', 'OP_CARRIER', 'TAIL_NUM']
numeric_columns = [col for col in numeric_columns if col not in columns_to_remove]

"""joined3M = joined3M.filter(joined3M['CANCELLED'] < 1)[string_columns + numeric_columns + date_columns].dropna()

joined3M.cache()  # Cache the DataFrame for faster access"""

# COMMAND ----------

# Select relevant columns
selected_cols = string_columns + numeric_columns + date_columns
relev_data = joined3M.select(*selected_cols)

# can't have nulls in label column
relev_data = relev_data.dropna(subset=['DEP_DEL15'])

#weird columns
relev_data = relev_data.drop('DEP_TIME_BLK', 'ARR_TIME_BLK', 'OP_CARRIER')
relev_data.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

# Create StringIndexer stages for each string column
indexers = [
    StringIndexer(inputCol=column, outputCol=f"{column}_index", handleInvalid='keep')
    for column in string_columns
]

# Apply StringIndexer transformations
indexer_models = [indexer.fit(relev_data) for indexer in indexers]

## is this needed?
for model in indexer_models:
    relev_data = model.transform(relev_data)
"""
# One-hot encoding indexed categorical columns
encoder = OneHotEncoder(
    inputCols=[f"{column}_index" for column in string_columns],
    outputCols=[f"{column}_vec" for column in string_columns]
)"""

encoder = OneHotEncoder(
    inputCols=[indexer.getOutputCol() for indexer in indexers],
    outputCols=["{0}_vec".format(indexer.getOutputCol()) for indexer in indexers]
)

# Assemble indexed categorical columns and numerical columns
#input_cols = [f"{column}_vec" for column in string_columns] + numeric_columns
input_cols = encoder.getOutputCols() + numeric_columns
assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

# COMMAND ----------

columns_to_drop = [f"{column}_index" for column in string_columns]
relev_data = relev_data.drop(*columns_to_drop)

pipeline = Pipeline(stages=indexers + [encoder, assembler])
df_encoded = pipeline.fit(relev_data).transform(relev_data)

#df_encoded = df_encoded.drop(*string_columns)

# COMMAND ----------

df_encoded = df_encoded.drop(*string_columns)

# COMMAND ----------

# Create a pipeline combining indexing, assembling, and encoding stages
#or, try
#df_encoded = encoder.fit(relev_data).transform(relev_data)
#df_encoded = assembler.transform(df_encoded)

# COMMAND ----------

# DBTITLE 1,Train / Test Split
# repartition data
#df_encoded_sample = df_encoded_sample.repartition("sched_depart_date_time_UTC")

# Apply the percent_rank function within the partition
window_spec = Window.partitionBy().orderBy("sched_depart_date_time_UTC")
df_encoded = df_encoded.withColumn("rank", percent_rank().over(window_spec))

# Use the rank column to split the data into train and test sets
train_data = df_encoded.where("rank <= 0.8").drop("rank")
test_data = df_encoded.where("rank > 0.8").drop("rank")

# verify data is populating -- was having issues (comment out to save time)
#train_data.count()

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder

# Apply StringIndexer to 'OP_UNIQUE_CARRIER'
string_indexer = StringIndexer(inputCol='OP_UNIQUE_CARRIER', outputCol='OP_UNIQUE_CARRIER_index')
indexed = string_indexer.fit(relev_data).transform(relev_data)

# Apply OneHotEncoder to 'OP_UNIQUE_CARRIER_index'
encoder = OneHotEncoder(inputCol='OP_UNIQUE_CARRIER_index', outputCol='OP_UNIQUE_CARRIER_encoded')
encoded = encoder.fit(indexed).transform(indexed)

# Show the resulting encoded column
encoded.select('OP_UNIQUE_CARRIER', 'OP_UNIQUE_CARRIER_encoded').show(truncate=False)


# COMMAND ----------

# DBTITLE 1,Train DT
# Define the models
dt = DecisionTreeClassifier(labelCol='DEP_DEL15', featuresCol='features')
dt_model = dt.fit(train_data)

# Create a pipeline for the Decision Tree on the training data
#pipeline_dt = Pipeline(stages=indexers + [encoder, assembler, dt])
#model_dt = pipeline_dt.fit(train_data)

# COMMAND ----------

# DBTITLE 1,Predictions
# Make predictions on the test data using Decision Tree
predictions_dt = dt_model.transform(test_data)
predictions_dt.select('DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'MONTH', 'DEP_DELAY','DISTANCE', 'origin_DailyPrecipitation').show(10)

# COMMAND ----------

# DBTITLE 1,Feature Importance
feature_importance = model_dt.featureImportances

for idx, feature in enumerate(feature_columns):
    print(f"Feature '{feature}' has importance: {feature_importance[idx]}")

# COMMAND ----------

# DBTITLE 1,Evaluation Metrics
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions_dt)
recall = evaluator.evaluate(predictions_dt, {evaluator.metricName: "weightedRecall"})
precision = evaluator.evaluate(predictions_dt, {evaluator.metricName: "weightedPrecision"})
f1_score = evaluator.evaluate(predictions_dt, {evaluator.metricName: "f1"})

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1_score)

# COMMAND ----------

# DBTITLE 1,Plot decision tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
class_names = ['delayed', 'not delayed']

# visualise the decision tree
fig = plt.figure(figsize=(14,8))
_ = plot_tree(model_dt, 
              feature_names=feature_columns,
              filled=True,
              class_names=class_names,
              fontsize=10)

# COMMAND ----------

# look at the feature importances
dfFeatures = pd.DataFrame({'Features':feature_columns.tolist(),'Importances':model_dt.feature_importances_})
dfFeatures.sort_values(by='Importances',ascending=False).head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

# DBTITLE 1,Train, Predictions for RF
# Define Random Forest Classifier and model
rf = RandomForestClassifier(labelCol='DEP_DEL15', featuresCol='features')
rfModel = rf.fit(train_df)

# Create a pipeline for the Random Forest on the training data
pipeline_rf = Pipeline(stages=indexers + [encoder, assembler, rf])
model_rf = pipeline_rf.fit(train_data)

# Make predictions on the test data using Random Forest
predictions_rf = model_rf.transform(test_data)
predictions_rf.select('DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'MONTH', 'DEP_DELAY','DISTANCE', 'origin_DailyPrecipitation', 'probability').show(10)

# COMMAND ----------

# DBTITLE 1,Evaluation Metrics
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions_rf)
recall = evaluator.evaluate(predictions_rf, {evaluator.metricName: "weightedRecall"})
precision = evaluator.evaluate(predictions_rf, {evaluator.metricName: "weightedPrecision"})
f1_score = evaluator.evaluate(predictions_rf, {evaluator.metricName: "f1"})

print("Accuracy: ", accuracy)
print("Recall: ", recall)
print("Precision: ", precision)
print("F1-Score: ", f1_score)

# COMMAND ----------

# DBTITLE 1,Grid Search Cross Val
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()

# Define the parameter grid to search through
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500]  # can play around with this
}

# GridSearch cross validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')

# Separate features (X) and target variable (y)
X_train = train_df.drop('target', axis=1)  # Features for training
y_train = train_df['target']  # Target variable for training

# Separate test features (X_test) and target variable (y_test)
X_test = test_df.drop('target', axis=1)  # Features for testing
y_test = test_df['target']  # Target variable for testing

# COMMAND ----------

# DBTITLE 1,Random Forest: Best Parameters
# Fit the model and find the best parameters
## train
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_scores = grid_search.cv_results_

print("Best Parameters:", best_params)
print("Best Scores:", best_scores)


## retrain classifier using the best params
best_rf_classifier = RandomForestClassifier(**best_params)
best_rf_classifier.fit(X_train, y_train)

# Evaluate the model on the test dataset
test_accuracy = best_rf_classifier.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# COMMAND ----------

# DBTITLE 1,Try Different numTrees


# COMMAND ----------

########################

# COMMAND ----------

# DBTITLE 1,Understanding RF Probabilities
rf_predictions.select("probability").show(truncate=False)

# first prob = probability that the class is the most frequent class in the train set 
# second prob = probability that the class is the less frequent class in the train set

# in other words:
    #['My confidence that the predicted label = the true label', 'My confidence that the label != the true label']

# COMMAND ----------

# DBTITLE 1,Art Extract Prob
from pyspark.sql.types import DoubleType, ArrayType
def extract_prob(v):
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None

extract_prob_udf = udf(extract_prob, DoubleType())
predictions = rf_predictions.withColumn("prob_pos", extract_prob_udf(col("probability")))

# COMMAND ----------

# DBTITLE 1,Art Check Prob Thresholds
# Set decison cut offs
CutOffs = [0, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80]

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
results_pd
