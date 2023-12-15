# Databricks notebook source
# MAGIC %md
# MAGIC # Big Picture Visuals

# COMMAND ----------

# analysis requirements
from Code.funcs import blob_connect
import pandas as pd
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
import plotly.express as px
import mlflow

# COMMAND ----------

team_blob_url = blob_connect()
df = spark.read.parquet(f"{team_blob_url}/LH/1yr_clean_temp_2")
df = df.dropna()

# COMMAND ----------

import matplotlib.pyplot as plt

# Calculating count of delayed and not delayed flights
delayed_flights = df.filter(df['DEP_DEL15'] == 1).count()
not_delayed_flights = df.filter(df['DEP_DEL15'] == 0).count()

# Creating a bar plot for delayed vs. not delayed flights
plt.figure(figsize=(6, 4))
plt.bar(['Delayed', 'Not Delayed'], [delayed_flights, not_delayed_flights], color=['red', 'green'])
plt.xlabel('Flight Delay')
plt.ylabel('Count')
plt.title('Delayed vs. Not Delayed Flights')
plt.show()


# COMMAND ----------

from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your Spark DataFrame with numerical columns
numerical_columns = [col for col, dtype in df.dtypes if dtype in ['int', 'double']]

# Get numerical features into a single vector
assembler = VectorAssembler(inputCols=numerical_columns, outputCol="features")
df_assembled = assembler.transform(df).select("features")

# Apply PCA
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
model = pca.fit(df_assembled)
transformed = model.transform(df_assembled).select("pca_features")

# Extracting PCA components to Pandas DataFrame for visualization
pandas_df = transformed.select("pca_features").toPandas()

# Splitting the dense vector into separate columns
pandas_df[['PC1', 'PC2']] = pd.DataFrame(pandas_df['pca_features'].tolist(), index=pandas_df.index)

# Plotting PCA components
plt.figure(figsize=(8, 6))
plt.scatter(pandas_df['PC1'], pandas_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()


# COMMAND ----------

# Apply PCA
pca = PCA(k=5, inputCol="features", outputCol="pca_features")
model = pca.fit(df_assembled)
transformed = model.transform(df_assembled).select("pca_features")

# Extracting PCA components to Pandas DataFrame for visualization
pandas_df = transformed.select("pca_features").toPandas()

# Splitting the dense vector into separate columns
pandas_df[['PC1', 'PC2']] = pd.DataFrame(pandas_df['pca_features'].tolist(), index=pandas_df.index)

# Plotting PCA components
plt.figure(figsize=(8, 6))
plt.scatter(pandas_df['PC1'], pandas_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()


# COMMAND ----------


# Assuming 'df' is your Spark DataFrame with numerical columns
numerical_columns = [col for col, dtype in df.dtypes if dtype in ['int', 'double']]

# Get numerical features into a single vector
assembler = VectorAssembler(inputCols=numerical_columns, outputCol="features")
df_assembled = assembler.transform(df).select("features")

# Apply PCA
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
model = pca.fit(df_assembled)
transformed = model.transform(df_assembled).select("pca_features")

# Plotting PCA components using Plotly
plot_data = transformed.select("pca_features").toPandas()

# Log plot to Spark DataBricks
with mlflow.start_run() as run:
    fig = px.scatter(plot_data, x='pca_features[0]', y='pca_features[1]', title='PCA Visualization')
    display(fig)


# COMMAND ----------

################################################################################################################################################################

# COMMAND ----------

# MAGIC %md
# MAGIC # Decision Trees and Random Forest

# COMMAND ----------

# DBTITLE 1,Imports
# analysis requirements
import pandas as pd
import numpy as np
from pyspark.sql.functions import udf, isnan, when, count, col, expr, regexp_extract
from pyspark.sql.types import StringType, IntegerType, FloatType, BooleanType
import pyspark.sql.functions as F
from pyspark.sql import types
import re
import matplotlib.pyplot as plt

# spark
from pyspark import SparkContext
from pyspark.sql import SparkSession

# cross val 
import statsmodels.api as sm
from pyspark.sql.functions import monotonically_increasing_id
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV

# log regression, decision tree, random forest
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import VectorAssembler, StringIndexer
from sklearn.tree import DecisionTreeRegressor

# evaluation 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, confusion_matrix, classification_report
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# COMMAND ----------

# get storage
mids261_mount_path = "/mnt/mids-w261"

# load 3-month data with datatypes
df_combined_3 = spark.read.load(f"{mids261_mount_path}/OTPW_3M_2015.csv",format="csv", inferSchema="true", header="true")

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

# get visual of nulls
"""
summary_stats = df_combined_3.select(drop_cols).describe().toPandas().transpose()
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.axis('off') 
tbl = ax.table(cellText=summary_stats.values, colLabels=summary_stats.columns, loc='center')
plt.title('Features with > 90% Nulls')
plt.xlabel('Feature')
plt.ylabel('Values')
plt.tight_layout()
plt.show()
plt.savefig('summary_statistics.png', bbox_inches='tight', pad_inches=0.1, transparent=True, format='png')
"""

#drop columns from dataframe
df_combined_3 = df_combined_3.drop(*drop_cols)
df_combined_3.display()

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
# MAGIC ## Cross-Validation: Test / Train Split

# COMMAND ----------

train, test = assembled_df.randomSplit([0.7, 0.3], seed = 42)
kfold = StratifiedKFold(n_splits=10)

lr_model = LogisticRegressionCV(featuresCol='features', labelCol='label', cv=kfold,class_weight='balanced',random_state=888)
model = lr_model.fit(train)
model

# COMMAND ----------

# DBTITLE 0,Cross-Validation: Test / Train Split
######## Can't do random because of time series
#train, test = assembled_df.randomSplit([0.7, 0.3], seed = 42)

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


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Feature Selection p2: Decision Tree

# COMMAND ----------

# Use assembler from Random Forest
feature_columns = numeric_features
#df = assembler.transform(df_combined_3) -> already did this from RF. need to clean that code up now

# decision tree classifier and training
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
dt_model = dt.fit(cv_train)

# Make predictions
dt_predictions = dt_model.transform(cv_test)

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
# MAGIC ## Logistic Regression

# COMMAND ----------

from sklearn.feature_selection import RFE

# logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='label')
model = lr.fit(cv_train)

# Use RFE to select the top 10 features
rfe = RFE(model, n_features_to_select=10)
selected_features = cv_train[:, rfe.support_]
print(selected_features)

# Make predictions on the test set
lr_predictions = model.transform(cv_test)
lr_predictions.select(selected_features).show(25)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Random Forest

# COMMAND ----------

# Define Random Forest Classifier and model
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(cv_train)

# Look at some predictions
rf_predictions = rfModel.transform(cv_test)
rf_predictions.select('DAY_OF_WEEK', 'OP_CARRIER_FL_NUM', 'HourlyDewPointTemperature', 'HourlyRelativeHumidity', 'label', 'ORIGIN_AIRPORT_ID', 'prediction', 'probability').show(25)


# COMMAND ----------

# DBTITLE 1,Evaluation and Predictions
# Accuracy
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator_acc.evaluate(rf_predictions)

# Precision
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator_precision.evaluate(rf_predictions)

# Recall
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator_recall.evaluate(rf_predictions)

# Print metrics
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"Accuracy = {accuracy}")
print(f"Test Error = ", 1.0 - accuracy)

# Compare the actual values and predicted values
rf_predictions.select("label", "prediction").show(10)

# COMMAND ----------

########################################################################

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
