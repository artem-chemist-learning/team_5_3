# Databricks notebook source
# MAGIC %md
# MAGIC #### Imports, Load in Data

# COMMAND ----------

# analysis requirements
import pandas as pd
import numpy as np
from pyspark.sql.functions import udf, isnan, when, count, col, expr, regexp_extract
from pyspark.sql.types import StringType, IntegerType, FloatType, BooleanType
import pyspark.sql.functions as F
from pyspark.sql import types
import re

# spark
from pyspark import SparkContext
from pyspark.sql import SparkSession

# random forest
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import VectorAssembler, StringIndexer
from sklearn.tree import DecisionTreeRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

# get storage
mids261_mount_path = "/mnt/mids-w261"

# load 3-month data with datatypes
df_combined_3 = spark.read.load(f"{mids261_mount_path}/OTPW_3M_2015.csv",format="csv", inferSchema="true", header="true")

# COMMAND ----------

df_combined_3.display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Feature Selection p1: Drop Nulls

# COMMAND ----------

# get null counts
null_vals = df_combined_3.select([count(when(col(c).isNull(), c)).alias(c) for c in df_combined_3.columns])

# null percents
data_size = int(df_combined_3.count())
null_percents = df_combined_3.select([(100.0 * count(when(col(c).isNull(), c))/data_size).alias(c) for c in df_combined_3.columns])
null_per_t = null_percents.toPandas().T.reset_index(drop=False)
null_per_t = null_per_t[null_per_t[0] > 90]

# columns to drop based on >90% nulls
drop_cols = null_per_t['index'].tolist()
df_combined_3 = df_combined_3.drop(*drop_cols)
df_combined_3.display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Transform Remaining Features

# COMMAND ----------

# DBTITLE 1,Impute Data
# MAGIC %md
# MAGIC ####Features that need transformation
# MAGIC https://medium.com/airbnb-engineering/overcoming-missing-values-in-a-random-forest-classifier-7b1fc1fc03ba#.1104o9tnm
# MAGIC "fill in missing values with the median (for numerical values) or mode (for categorical values"
# MAGIC
# MAGIC ###### Delays
# MAGIC   - CARRIER_DELAY: null 
# MAGIC   - WEATHER_DELAY: null 
# MAGIC   - etc. etc.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data for DecisionTree, RandomForest

# COMMAND ----------

# DBTITLE 1,Cast Values, Numeric Features
# Convert string to float for random forest use -> these are manual right now
cast_features = ["QUARTER", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID"]

# casting daily features as int for random forest classifier 
for col_name in cast_features:
    df_combined_3 = df_combined_3.withColumn(col_name, col(col_name).cast('int'))

# Select numeric feature columns
numeric_features = [t[0] for t in df_combined_3.dtypes if t[1] == 'int' or t[1] == 'float']
df_combined_3.select(numeric_features).describe().toPandas().transpose() ## this is a helpful display

# COMMAND ----------

# DBTITLE 1,Vector Assembler & Features
## Drop features column if it already exsits
if 'features' in df_combined_3.columns:
    df_combined_3 = df_combined_3.drop('features')

# Combine features into a single vector column using vector assembler
assembler = VectorAssembler(inputCols=numeric_features, outputCol="features", handleInvalid="skip")
assembled_df = assembler.transform(df_combined_3)

assembled_df.select("features").display(truncate=False)

# COMMAND ----------

# DBTITLE 1,Add 'label' column: Target predictor variable
# Calculate time difference in minutes between actual departure time and scheduled departure time (and convert to minutes)
time_difference = (col("DEP_TIME") - col("CRS_DEP_TIME")) / 60 

# Create a new column indicating whether the flight was delayed by 15+ minutes within 2 hours of departure
assembled_df = assembled_df.withColumn("TIME_DIFFERENCE", time_difference)
assembled_df = assembled_df.withColumn(
    "label", 
    expr("CASE WHEN (DEP_DELAY >= 15 OR DEP_DEL15 == 1) AND (TIME_DIFFERENCE <= 120 AND TIME_DIFFERENCE >= 0) THEN 1 ELSE 0 END")
)

pd.DataFrame(assembled_df.take(110), columns=assembled_df.columns).transpose()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Feature Selection p2: Decision Tree

# COMMAND ----------

# DBTITLE 1,Test / Train Split
######## Can't do random because of time series. Will change later
train, test = assembled_df.randomSplit([0.7, 0.3], seed = 42)

# COMMAND ----------

# Use assembler from Random Forest
feature_columns = numeric_features
#df = assembler.transform(df_combined_3) -> already did this from RF. need to clean that code up now

# decision tree classifier and training
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
dt_model = dt.fit(train)

# Make predictions
dt_predictions = dt_model.transform(test)

# Evaluation metrics
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(dt_predictions)
print(f"Accuracy: {accuracy}")

# Display the decision tree
print("Learned classification tree model:")
tree_model = dt_model.toDebugString
print(tree_model)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Random Forest

# COMMAND ----------

# DBTITLE 1,Train RF and Get Preds
# Define Random Forest Classifier and model
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(train)

# Look at some predictions
predictions = rfModel.transform(test)
predictions.select('DAY_OF_WEEK', 'OP_CARRIER_FL_NUM', 'HourlyDewPointTemperature', 'HourlyRelativeHumidity', 'label', 'ORIGIN_AIRPORT_ID', 'prediction', 'probability').show(25)


# COMMAND ----------

# DBTITLE 1,Evaluation and Predictions
# Accuracy
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator_acc.evaluate(predictions)

# Precision
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator_precision.evaluate(predictions)

# Recall
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator_recall.evaluate(predictions)

# Print metrics
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"Accuracy = {accuracy}")
print(f"Test Error = ", 1.0 - accuracy)

# Compare the actual values and predicted values
predictions.select("label", "prediction").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Gradient Boosting

# COMMAND ----------

############## In progress ##############

# GET MEAN VALUES OF Y (from Vini demo)
#selected_df["y"] = pd.to_numeric(df["y"])
#mean_y = df.groupby('x').mean().reset_index().sort_values(by=['y'])
#mean_y

# Function from Vini's Demo 11
def GradientBoosting(nTrees, nDepth, gamma, bagFrac, X, Y):
    nDataPts = len(X)
    nSamp = int(bagFrac * nDataPts)
    
    # Define function T to accumulate average prediction functions from trained trees.  
    # initialize T to fcn mapping all x to zero to start 
    T = lambda x: 0.0
    
    # loop to generate individual trees in ensemble
    for i in range(nTrees):
        
        # take a random sample from the data
        sampIdx = np.random.choice(nDataPts, nSamp)
        
        xTrain = X[sampIdx]
        
        # estimate the regression values with the current trees.  
        yEst = T(xTrain)
        
        # subtract the estimate based on current ensemble from the labels
        yTrain = Y[sampIdx] - np.array(yEst).reshape([-1,1])
        
        # build a tree on the sampled data using residuals for labels
        tree = DecisionTreeRegressor(max_depth=nDepth)
        tree.fit(xTrain, yTrain)
                
        # add the new tree with a learning rate parameter (gamma)
        T = wt_sum_fcn(T, tree.predict, gamma)
    return T

# COMMAND ----------

nTrees = 10  # try changing the number of trees being built
nDepth = 3   # fairly deep for 100 data points
gamma = 0.1
bagFrac = 1   # Bag fraction - how many points in each of the random subsamples.  

gbst = GradientBoosting(nTrees, nDepth, gamma, bagFrac, X, Y)

result = gbst(X)

plt.plot(X, result, 'r')
plt.scatter(X,Y)
display(plt.show())

# COMMAND ----------


########## IN PROGRESS ##########
## Try doing it with a Spark DataFrame
# Convert DataFrame to LabeledPoint rdd (need label, features columns in rdd)
rdd_3_month = df_combined_3.rdd.map(lambda row: (row['label'], row['features']))
rdd_labeled = rdd_3_month.map(lambda x: LabeledPoint(x[0], x[1]))

# Set params
numClasses = 2  
categoricalFeaturesInfo = {} 
numTrees = 3 
seed = 42

# Model
model = RandomForest.trainClassifier(rdd_labeled, numClasses, categoricalFeaturesInfo,
                                     numTrees, seed=seed)

print(model)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Regularization
# MAGIC TODO

# COMMAND ----------

# MAGIC %md
# MAGIC #### References
# MAGIC 1. https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.tree.RandomForest.html
# MAGIC 2. https://www.analyticsvidhya.com/blog/2021/05/feature-transformations-in-data-science-a-detailed-walkthrough/
# MAGIC 3. https://spark.apache.org/docs/latest/ml-features
# MAGIC 4. https://towardsdatascience.com/a-guide-to-exploit-random-forest-classifier-in-pyspark-46d6999cb5db
# MAGIC 5. https://chat.openai.com/
# MAGIC
