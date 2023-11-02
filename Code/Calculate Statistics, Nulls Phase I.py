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
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import sum,avg,max,count
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

# Check datatypes of data
df_combined_3.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Take a closer look at the columns
# MAGIC Split the columns into numerical and categorial so that we can calculate statistics.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get mean, std dev, ranges on numeric columns
