# Databricks notebook source
# MAGIC %md
# MAGIC # Decision Trees and Random Forest

# COMMAND ----------

# DBTITLE 1,Imports
# analysis requirements
from Code.funcs import blob_connect
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

# DBTITLE 1,Load Dataset
# read in daily weather data from parquet
team_blob_url = blob_connect()
joined3M = spark.read.parquet(f"{team_blob_url}/ES/new_joins/3MO_schema")

# Take only data needed
#take only columns needed
df_clean = joined3M.dropna()

#df_clean = df_clean.withColumn('sched_depart_date_time_UTC', to_timestamp(df_clean['sched_depart_date_time_UTC']))
#df_clean = df_clean.withColumn('DEP_DELAY', df_clean['DEP_DELAY'].cast(DoubleType()))

# COMMAND ----------

import sys
sys.path.append("/Workspace/artem.lebedev@berkeley.edu/team_5_3/Code")
from funcs.py import describe_table
from Logistic_regression import 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree

# COMMAND ----------

# DBTITLE 1,Vector Assembler & Features


# COMMAND ----------


