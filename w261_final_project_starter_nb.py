# Databricks notebook source
from pyspark.sql.functions import col
print("Welcome to the W261 final project!") 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Know your mount
# MAGIC Here is the mounting for this class, your source for the original data! Remember, you only have Read access, not Write! Also, become familiar with `dbutils` the equivalent of `gcp` in DataProc

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

dbutils.fs.help()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data for the Project
# MAGIC
# MAGIC For the project you will have 4 sources of data:
# MAGIC
# MAGIC 1. Airlines Data: This is the raw data of flights information. You have 3 months, 6 months, 1 year, and full data from 2015 to 2019. Remember the maxima: "Test, Test, Test", so a lot of testing in smaller samples before scaling up! Location of the data? `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/`, `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_1y/`, etc. (Below the dbutils to get the folders)
# MAGIC 2. Weather Data: Raw data for weather information. Same as before, we are sharing 3 months, 6 months, 1 year
# MAGIC 3. Stations data: Extra information of the location of the different weather stations. Location `dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/`
# MAGIC 4. OTPW Data: This is our joined data (We joined Airlines and Weather). This is the main dataset for your project, the previous 3 are given for reference. You can attempt your own join for Extra Credit. Location `dbfs:/mnt/mids-w261/OTPW_60M/` and more, several samples are given!

# COMMAND ----------

# Airline Data    
df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/")
display(df_flights)

# COMMAND ----------

# Weather data
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")
display(df_weather)

# COMMAND ----------

# Stations data      
df_stations = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")
display(df_stations)

# COMMAND ----------

# OTPW
df_otpw = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
display(df_otpw)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Example of EDA

# COMMAND ----------

import pyspark.sql.functions as F
import matplotlib.pyplot as plt

df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/")

# Grouping and aggregation for df_stations
grouped_stations = df_stations.groupBy('neighbor_id').agg(
    F.avg('distance_to_neighbor').alias('avg_distance_to_neighbor'),
).orderBy('avg_distance_to_neighbor')

display(grouped_stations)

# Grouping and aggregation for df_flights
grouped_flights = df_flights.groupBy('OP_UNIQUE_CARRIER').agg(
    F.avg('DEP_DELAY').alias('Avg_DEP_DELAY'),
    F.avg('ARR_DELAY').alias('Avg_ARR_DELAY'),
    F.avg('DISTANCE').alias('Avg_DISTANCE')
)

display(grouped_flights)

# Convert columns to appropriate data types
df_weather = df_weather.withColumn("HourlyPrecipitationDouble", F.col("HourlyPrecipitation").cast("double"))
df_weather = df_weather.withColumn("HourlyVisibilityDouble", F.col("HourlyVisibility").cast("double"))
df_weather = df_weather.withColumn("HourlyWindSpeedDouble", F.col("HourlyWindSpeed").cast("double")).filter(col("HourlyWindSpeedDouble") < 2000)

# Overlayed boxplots for df_weather
weather_cols = ['HourlyPrecipitationDouble', 'HourlyVisibilityDouble', 'HourlyWindSpeedDouble']
weather_data = df_weather.select(*weather_cols).toPandas()

plt.figure(figsize=(10, 6))
weather_data.boxplot(column=weather_cols)
plt.title('Boxplots of Weather Variables')
plt.xlabel('Weather Variables')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Pipeline Steps For Classification Problem
# MAGIC
# MAGIC These are the "normal" steps for a Classification Pipeline! Of course, you can try more!
# MAGIC
# MAGIC ## 1. Data cleaning and preprocessing
# MAGIC
# MAGIC * Remove outliers or missing values
# MAGIC * Encode categorical features
# MAGIC * Scale numerical features
# MAGIC
# MAGIC ## 2. Feature selection
# MAGIC
# MAGIC * Select the most important features for the model
# MAGIC * Use univariate feature selection, recursive feature elimination, or random forest feature importance
# MAGIC
# MAGIC ## 3. Model training
# MAGIC
# MAGIC * Train a machine learning model to predict delays more than 15 minutes
# MAGIC * Use logistic regression, decision trees, random forests, or support vector machines
# MAGIC
# MAGIC ## 4. Model evaluation
# MAGIC
# MAGIC * Evaluate the performance of the trained model on a holdout dataset
# MAGIC * Use accuracy, precision, recall, or F1 score
# MAGIC
# MAGIC ## 5. Model deployment
# MAGIC
# MAGIC * Deploy the trained model to a production environment
# MAGIC * Deploy the model as a web service or as a mobile app
# MAGIC
# MAGIC ## Tools
# MAGIC
# MAGIC * Spark's MLlib and SparkML libraries
# MAGIC * These libraries have parallelized methods for data cleaning and preprocessing, feature selection, model training, model evaluation, and model deployment which we will utilize for this classification problem.
# MAGIC

# COMMAND ----------


