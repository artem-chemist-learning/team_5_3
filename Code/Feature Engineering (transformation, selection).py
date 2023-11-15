# Databricks notebook source
# MAGIC %md
# MAGIC #### Imports, Load in Data

# COMMAND ----------

# analysis requirements
import pandas as pd
import numpy as np
from pyspark.sql.functions import udf, isnan, when, count, col, regexp_extract
from pyspark.sql.types import StringType, IntegerType, FloatType, BooleanType
import pyspark.sql.functions as F
from pyspark.sql import types
import re

# spark
from pyspark import SparkContext
from pyspark.sql import SparkSession

# random forest
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils

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
# MAGIC #### Feature Selection p1: Drop Nulls

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
# MAGIC #### Transform Remaining Features

# COMMAND ----------

## Features that need transformation
#TODO

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Feature Selection p2: Random Forest

# COMMAND ----------

# Spark session
#spark = SparkSession.builder.appName("ExtractLabelFeatures").getOrCreate()
#sc = SparkContext("local", "RandomForestData")

# Extract feature columns from data
# TODO

# Convert DataFrame to LabeledPoint rdd (need label, features columns in rdd)
rdd_3_month = df_combined_3.rdd.map(lambda row: (row['label'], row['features']))
rdd_labeled = rdd_3_month.map(lambda x: LabeledPoint(x[0], x[1]))

# Set params
numClasses = 2  # change?
categoricalFeaturesInfo = {}  # TODO
numTrees = 3  # change?
seed = 42

# Model
model = RandomForest.trainClassifier(rdd_labeled, numClasses, categoricalFeaturesInfo,
                                     numTrees, seed=seed)


# COMMAND ----------

# Random Forest model
print("Num trees: ", model.numTrees())
print("Num nodes: ", model.totalNumNodes())
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
# MAGIC
# MAGIC
