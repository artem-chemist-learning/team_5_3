# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Technical notebook to do trivial LR

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
from pyspark.sql.types import IntegerType, FloatType, DoubleType,  ArrayType
from pyspark.sql.functions import size, to_timestamp, mean as _mean, stddev as _stddev, col, sum as _sum, rand, when, collect_list, udf, date_trunc, count, lag, first, last, percent_rank, array
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, IndexToString
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression as LR

team_blob_url = blob_connect()

# COMMAND ----------

# Read from storage
joined = spark.read.parquet(f"{team_blob_url}/BK/clean_5yr_WITHOUT_2019_eng")

# Split into train - validations 
number_of_folds = 1
test_train_sets = create_validation_blocks(joined, "sched_depart_date_time_UTC", block = number_of_folds, split=0.8)
print(f"Train count     : {test_train_sets[0][0].count()}")
print(f"Validation count:  {test_train_sets[0][1].count()}")

# COMMAND ----------

test_train_sets[0][1].count()

# COMMAND ----------

# Write train into blob
test_train_sets[0][0].write.mode("overwrite").parquet(f"{team_blob_url}/BK/pure_train")

# COMMAND ----------

# Write validation into blob
test_train_sets[0][1].write.mode("overwrite").parquet(f"{team_blob_url}/BK/pure_val")

# COMMAND ----------


