# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Getting nulls, statistics: Phase I

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Import packages and load datasets

# COMMAND ----------

# Import statements
import pandas as pd
import numpy as np
from pyspark.sql.functions import udf, isnan, when, count, col, regexp_extract
from pyspark.sql.types import StringType, IntegerType, FloatType, BooleanType
import pyspark.sql.functions as F
from pyspark.sql import types
import re
#import sum,avg,max,count

#set path 
mids261_mount_path = "/mnt/mids-w261"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Combined Dataset

# COMMAND ----------

# Read in combined dataset, 3 months
df_combined_3 = spark.read.format("csv") \
                     .option("header","true") \
                     .load(f"dbfs:{mids261_mount_path}/OTPW_3M_2015.csv")
df_combined_3.display()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Nulls, missing data

# COMMAND ----------

# First, we take a look at how many nulls are in each column
# nulls
null_vals = df_combined_3.select([count(when(col(c).isNull(), c)).alias(c) for c in df_combined_3.columns])
null_vals.display()

# COMMAND ----------

# Let's calculate percentage of nulls for each field, given the nulls and count of each field
data_size = int(df_combined_3.count())

null_percents = df_combined_3.select([(100.0 * count(when(col(c).isNull(), c))/data_size).alias(c) for c in df_combined_3.columns])
null_percents.display()

# COMMAND ----------

# Filtering out columns where there were more than 90% of the data missing
null_per_t = null_percents.toPandas().T.reset_index(drop=False)
null_per_t = null_per_t[null_per_t[0] > 90]
null_per_t

# COMMAND ----------

# Lastly, we will drop columns that have too many nulls so that we can ignore them in casting
drop_cols = null_per_t['index'].tolist()
drop_cols

# COMMAND ----------

# As one extra step, we can also check for nans, but there don't appear to be any
df_combined_3.select([count(when(isnan(c), c)).alias(c) for c in df_combined_3.columns]).display()

# COMMAND ----------

# Now, we drop the columns with too many nulls
df_combined_3 = df_combined_3.drop(*drop_cols)
df_combined_3.display()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Cast correct datatypes

# COMMAND ----------

# Function to try casting as detected datatype
def cast_dtype(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

# COMMAND ----------

# Recast date
df_combined_3 = df_combined_3.withColumn("DATE", 
                                  F.concat(F.lit("2015-"),
                                  F.col("DAY_OF_MONTH"), F.lit("-"),
                                  F.col("DAY_OF_WEEK")).cast(types.TimestampType()))

# Call casting function
#def get_rows(row):
#    #return row["FL_DATE"]
#    print(row["FL_DATE"])
#df_combined_3.foreach(get_rows)

# COMMAND ----------

# Define the UDF (User-Defined Function) to apply your casting function
cast_dtype_udf = udf(cast_dtype, StringType())

# Create a new column with the casted values
df_combined_3 = df_combined_3.withColumn("CASTED_latt", cast_dtype_udf(df_combined_3["origin_airport_lat"]))

# Show the DataFrame
df_combined_3.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Take a closer look at the columns
# MAGIC Split the columns into numerical and categorial so that we can calculate statistics.

# COMMAND ----------

# Check if the first character is a digit
first_record = df_combined_3.collect()[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get mean, std dev, ranges on numeric columns

# COMMAND ----------

 df_combined_3.describe().display()
