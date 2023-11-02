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
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType, FloatType
import pyspark.sql.functions as F
from pyspark.sql import types
#import sum,avg,max,count

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Data

# COMMAND ----------

# Read full weather dataset
mids261_mount_path = "/mnt/mids-w261"
df_weather = spark.read.parquet(f"dbfs:{mids261_mount_path}/datasets_final_project_2022/parquet_weather_data/")
df_weather.display()

# 3 month weather
df_weather_3 = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Airline Data

# COMMAND ----------

# Read full airline dataset
df_flights = spark.read.parquet(f"dbfs:{mids261_mount_path}/datasets_final_project_2022/parquet_airlines_data/")
df_flights.display()

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

#df_combined_3.show(df_combined_3.count(), truncate = True)
df_combined_3.describe().display()

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

# MAGIC %md
# MAGIC #### Get mean, std dev, ranges on numeric columns
