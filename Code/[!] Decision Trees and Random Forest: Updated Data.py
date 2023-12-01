# Databricks notebook source
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
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

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
joined3M = spark.read.parquet(f"{team_blob_url}/LH/1yr_clean_temp_2").cache()

## change to this later
#joined3M = spark.read.parquet(f"{team_blob_url}/LH/1yr_clean_temp").cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data for DecisionTree, RandomForest

# COMMAND ----------

# DBTITLE 1,Uses features we don't have access to 2 hrs before
# Select numeric feature columns
numeric_columns = [t[0] for t in joined3M.dtypes if t[1] == 'integer' or t[1] == 'float' or t[1] == 'double']

# drop features that conflict with predictor 
numeric_columns = [col for col in numeric_columns if col not in ['DEP_DELAY_NEW', 'DEP_DELAY', 'DEP_DEL15', 'CARRIER_DELAY', 'ARR_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'CANCELLED', 'WEATHER_DELAY', 'DIVERTED', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'DEP_TIME', 'CRS_DEP_TIME', 'TAXI_IN', 'TAXI_OUT']]

#predictor
label = 'DEP_DEL15'

#only numeric columns + label + splitting column
relev_data = joined3M.select(*numeric_columns, col("sched_depart_date_time_UTC"), col("DEP_DEL15").alias("label"))

#impute na's --> WAY too many get removed otherwise
relev_data = relev_data.dropna()
relev_data.count()

# COMMAND ----------

# DBTITLE 1,Vector Assembler & Features
# Combine features into a single vector column using vector assembler
assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features", handleInvalid="keep")
assembled_df = assembler.transform(relev_data)

assembled_df.select("features").display(truncate=False)

# COMMAND ----------

# DBTITLE 1,Test / Train Split
window_spec = Window.partitionBy().orderBy("sched_depart_date_time_UTC")
assembled_df = assembled_df.withColumn("rank", percent_rank().over(window_spec))

# Use the rank column to split the data into train and test sets
train_data = assembled_df.where("rank <= 0.8").drop("rank")
test_data = assembled_df.where("rank > 0.8").drop("rank")

# verify data is populating -- was having issues (comment out to save time)
#train_data.count()

# COMMAND ----------

# DBTITLE 1,Cross-Validation Split (not working)
"""
## add id to preserve rows
assembled_df = assembled_df.withColumn("row_id", monotonically_increasing_id())

# Calculate the number of rows per fold
n_splits = 4
total_rows = assembled_df.count()
rows_per_fold = total_rows // n_splits

# keep track of metrics
precision = []
recall = []
f_beta = []

# time series cross-validation
for i in range(n_splits):
    
    # start and end rows for each fold
    start_row = i * rows_per_fold
    end_row = start_row + rows_per_fold if i < n_splits - 1 else total_rows
    
    # split the data into train and test sets based on row_id
    cv_train = assembled_df.filter((assembled_df["row_id"] < start_row) | (assembled_df["row_id"] >= end_row))
    cv_test = assembled_df.filter((assembled_df["row_id"] >= start_row) & (assembled_df["row_id"] < end_row))
"""


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Decision Tree

# COMMAND ----------

# DBTITLE 1,Train DT (~2-5 min)
# decision tree classifier and training
dt = DecisionTreeClassifier(labelCol='label', featuresCol='features')
dt_model = dt.fit(train_data)

# COMMAND ----------

# DBTITLE 1,Get DT Output
# quick peek at the predictions
predictions_dt = dt_model.transform(test_data)
predictions_dt.select(numeric_columns[:6]).show(10)

# COMMAND ----------

# DBTITLE 1,Evaluate (~2-4 min)
# Precision
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator_precision.evaluate(predictions_dt)

# Recall
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator_recall.evaluate(predictions_dt)

# Print metrics
print(f"Precision: {precision}")
print(f"Recall:  {recall}")

# COMMAND ----------

# DBTITLE 1,Feature Importance
feature_importance = dt_model.featureImportances

for idx, feature in enumerate(numeric_columns):
    print(f"Feature '{feature}' has importance: {feature_importance[idx]}")

# COMMAND ----------

# DBTITLE 1,Read Tree
from pyspark.ml.classification import DecisionTreeClassificationModel

# Assuming dt_model is a trained DecisionTreeClassificationModel
if isinstance(dt_model, DecisionTreeClassificationModel):
    # Get the decision tree model's debug string
    tree_debug_string = dt_model.toDebugString

    # Display the decision tree rules
    print("Decision Tree Rules:")
    print(tree_debug_string)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC Note: Original experiment, only changed numTrees, rest was consistent --> no change in precision or recall
# MAGIC
# MAGIC Original Experiment
# MAGIC - Exp 1: numTrees=**25**, maxBins=default, maxDepth=default, minInstancesPerNode=default
# MAGIC - Exp 2: numTrees=**50**, maxBins=default, maxDepth=default, minInstancesPerNode=default
# MAGIC - Exp 3: numTrees=**100**, maxBins=default, maxDepth=default, minInstancesPerNode=default

# COMMAND ----------

# DBTITLE 1,Exp 1: numTrees=25, maxBins=10, maxDepth=5, minInstancesPerNode=1
rf_1 = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees=25, maxBins=10, maxDepth=5, minInstancesPerNode=1)
rfModel = rf_1.fit(train_data)

# COMMAND ----------

# DBTITLE 1,Exp 2: numTrees=25, maxBins=10, maxDepth=10, minInstancesPerNode=2
# Define Random Forest Classifier and model
rf_2 = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees=25, maxBins=10, maxDepth=10, minInstancesPerNode=2)
rfModel2 = rf_2.fit(train_data)

# COMMAND ----------

# DBTITLE 1,Exp 3: numTrees=10, maxBins=5, maxDepth=3, minInstancesPerNode=3
# Define Random Forest Classifier and model
rf_3 = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees=10, maxBins=50, maxDepth=3, minInstancesPerNode=3)
rfModel3 = rf_3.fit(train_data)

# COMMAND ----------

# DBTITLE 1,Get predictions
rf_1_predictions = rfModel.transform(test_data)
rf_2_predictions = rfModel2.transform(test_data)
rf_3_predictions = rfModel3.transform(test_data)

# COMMAND ----------

# DBTITLE 1,Experiment Results
# Evaluate predictions for each model
evaluator = MulticlassClassificationEvaluator(labelCol="label")

precision_1 = evaluator.evaluate(rf_1_predictions, {evaluator.metricName: "weightedPrecision"})
recall_1 = evaluator.evaluate(rf_1_predictions, {evaluator.metricName: "weightedRecall"})

precision_2 = evaluator.evaluate(rf_2_predictions, {evaluator.metricName: "weightedPrecision"})
recall_2 = evaluator.evaluate(rf_2_predictions, {evaluator.metricName: "weightedRecall"})

precision_3 = evaluator.evaluate(rf_3_predictions, {evaluator.metricName: "weightedPrecision"})
recall_3 = evaluator.evaluate(rf_3_predictions, {evaluator.metricName: "weightedRecall"})

# Create a table of precision and recall for each model
data = [
    ("Model 1", precision_1, recall_1),
    ("Model 2", precision_2, recall_2),
    ("Model 3", precision_3, recall_3)
]
columns = ["Model", "Precision", "Recall"]
precision_recall_df = spark.createDataFrame(data, columns)
precision_recall_df.display()

# COMMAND ----------

# DBTITLE 1,All Experiment Data
# add other metadata
num_trees_used = [25, 50, 10]

# Create a table of precision and recall for each model
#### !!!! update
model_data = [
    ("Model 1", precision_1, recall_1, 25, 10, 5, 1),
    ("Model 2", precision_2, recall_2, 50, 20, 10, 2),
    ("Model 3", precision_3, recall_3, 10, 5, 3, 3)
]
mod_columns = ["Model", "Precision", "Recall", "numTrees", "maxBins", "maxDepth", "minInstancesPerNode"]
model_df = spark.createDataFrame(model_data, mod_columns)
model_df.display()

# COMMAND ----------

# DBTITLE 1,Varying Precision with Different maxDepth
plt.figure(figsize=(8, 6))
model_pd = model_df.toPandas()

# Plot Precision
#plt.plot(model_pd['maxDepth'], model_pd['Precision'], marker='o', label=f'Precision - {Model}')

for model, data in model_pd.groupby('Model'):
    plt.plot(data['maxDepth'], data['Precision'], marker='o', label=f'Precision - {model}')

# Plot Recall
#plt.plot(model_pd['numTrees'], model_pd['Recall'], marker='o', label='Recall')

# Add labels, title, and legend
plt.xlabel('maxDepth')
plt.ylabel('Score')
plt.title('Precision for RF Models with Different maxDepth')
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Use best model to train RF (not working. slow)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Assuming rf_1 is defined as your RandomForestClassifier
# Set up the CrossValidator with parameters
paramGrid = (ParamGridBuilder()
             .addGrid(rf_1.numTrees, [10, 20, 30])
             .addGrid(rf_1.maxDepth, [5, 10, 15])
             .build())

evaluator = MulticlassClassificationEvaluator()

# Create CrossValidator
cross_val = CrossValidator(estimator=rf_1,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           numFolds=5)

# Fit the CrossValidator to the train_data
cv_model = cross_val.fit(train_data)

# Get the best model from CrossValidator
best_rf_model = cv_model.bestModel

# Make predictions on the test set using the best model
best_predictions = best_rf_model.transform(test_data)

# Evaluate the best model
best_precision = evaluator.evaluate(best_predictions, {evaluator.metricName: "weightedPrecision"})
print("Test Precision with best model: {:.2f}%".format(best_precision * 100))


# COMMAND ----------

# DBTITLE 1,Feature Importance (using Exp 1)
feature_importance_rf = rfModel.featureImportances

# Create a dictionary to store feature importance scores
importance_dict = {feature: importance for feature, importance in zip(numeric_columns, feature_importance_rf)}

# Sort the dictionary items by importance scores (descending order)
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print features sorted by importance
for feature, importance in sorted_importance:
    print(f"Feature '{feature}' has importance: {importance}")

# COMMAND ----------

# DBTITLE 1,Experiments: Vary maxDepth, maxBins (slow, ~25 minutes)
"""
# different parameters to try --> random for now
maxDepth_list = [5, 10, 15]
maxBins_list = [20, 30, 40]

results = []

for maxDepth in maxDepth_list:
    for maxBins in maxBins_list:
        # train Random Forest Classifier with specified parameters
        rf = RandomForestClassifier(featuresCol='features', labelCol='label', maxDepth=maxDepth, maxBins=maxBins)
        rfModel = rf.fit(train_data)
        predictions = rfModel.transform(test_data)
        
        # evaluate each model
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        
        # Store results in a dictionary
        result_dict = {
            "MaxDepth": maxDepth,
            "MaxBins": maxBins,
            "Precision": precision,
            "Recall": precision
        }
        
        # Append the dictionary to the results list
        results.append(result_dict)

# put results in a df
results_df = spark.createDataFrame(results)
results_df.display(truncate=False)
"""

# COMMAND ----------

# DBTITLE 1,[Art] Extract Probabilities, get metrics -- still needs work
from pyspark.sql.types import DoubleType, ArrayType
def extract_prob(v):
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None

extract_prob_udf = udf(extract_prob, DoubleType())
updated_predictions = rf_1_predictions.withColumn("prob_pos", extract_prob_udf(col("probability")))

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
updated_predictions = updated_predictions.withColumns({'TP':make_TP(updated_predictions.prob_pos, updated_predictions.label), 'FP':make_FP(updated_predictions.prob_pos, updated_predictions.label)})

# Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
num_cols = len(CutOffs)
TP_FP_pd = updated_predictions.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                        array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                        _sum(col("label")).alias("Positives")
                        )\
                        .toPandas()

# Convert the result into the pd df of precisions and recalls for each cu-off
results_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
results_pd['Precision'] = 100*results_pd['TP']/(results_pd['TP'] + results_pd['FP'])
results_pd['Recall']= 100*results_pd['TP']/TP_FP_pd.iloc[0,2]
results_pd

# COMMAND ----------

# MAGIC %md
# MAGIC #### References
# MAGIC 1. https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.tree.RandomForest.html
# MAGIC 2. https://www.analyticsvidhya.com/blog/2021/05/feature-transformations-in-data-science-a-detailed-walkthrough/
# MAGIC 3. https://spark.apache.org/docs/latest/ml-features
# MAGIC 4. https://towardsdatascience.com/a-guide-to-exploit-random-forest-classifier-in-pyspark-46d6999cb5db
# MAGIC 5. https://chat.openai.com/
# MAGIC
