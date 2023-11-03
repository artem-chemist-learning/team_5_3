# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Technical notebook to generate visuals for Phase I
# MAGIC
# MAGIC I am re-using Erik's code to look at the 3 month subset

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import sum,avg,max,count
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# COMMAND ----------

# MAGIC %md
# MAGIC #Read JFK weather data at daily level

# COMMAND ----------

# Read full dataset, get only daily data for NYC
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/")

# Take only weather at JFK
df_NYC_weather = df_weather.filter(df_weather['NAME'] == 'JFK INTERNATIONAL AIRPORT, NY US')
JFK_weather_daily = df_NYC_weather[['DATE','DailyAverageWindSpeed', 'DailyPrecipitation']].dropna().toPandas()

# Fix datatypes
JFK_weather_daily.columns = ['Date', 'Wind', 'Precipitation']
JFK_weather_daily['Date'] = pd.to_datetime(JFK_weather_daily['Date'], format = "%Y/%m/%d", utc=True, errors =  'coerce').dt.date
JFK_weather_daily['Wind'] = pd.to_numeric(JFK_weather_daily['Wind'], errors =  'coerce')
JFK_weather_daily['Precipitation'] = pd.to_numeric(JFK_weather_daily['Precipitation'], errors =  'coerce')
JFK_weather_daily.dropna(inplace = True)

JFK_weather_daily.head()


# COMMAND ----------

# MAGIC %md
# MAGIC #Read JFK flight data at daily level

# COMMAND ----------

# Read full dataset, get only daily data for NYC
df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/")

# Take only flights from JFK
JFK_codes = ['1247801', '1247802','1247803','1247804','1247805']
df_JFK_flights = df_flights.filter(df_flights['ORIGIN_AIRPORT_SEQ_ID'].isin(JFK_codes) )

# Aggregate delays to the daily level
delays_to_plot = df_JFK_flights[['FL_DATE','DEP_DELAY']].dropna()
JFK_delays_daily = delays_to_plot.groupBy('FL_DATE') \
                    .agg(avg('DEP_DELAY').alias("avg_delay"), \
                        max('DEP_DELAY').alias("max_delay") ) \
                    .toPandas()

# Convert to the right datatype, fix column names
JFK_delays_daily.columns = ['Date', 'Mean_delay', 'Max_delay']
JFK_delays_daily['Date'] = pd.to_datetime(JFK_delays_daily['Date'], format = "%Y/%m/%d", utc=True, errors =  'coerce').dt.date
JFK_delays_daily.dropna(inplace = True)
JFK_delays_daily.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #Merge data into one panda dateframe

# COMMAND ----------

df_to_plot = JFK_delays_daily.merge(JFK_weather_daily, on = 'Date')
df_to_plot.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #Make graph for raw data

# COMMAND ----------

# Instantiate figure and axis
num_rows = 3
num_columns = 1
fig, axes = plt.subplots(num_rows, num_columns, sharex=True)
fig.set_figheight(10)
#Adjust space between plots in the figure
#plt.subplots_adjust(hspace = 0.2)
mask = (df_to_plot['Date'] > pd.to_datetime('2015-10-01')) & (df_to_plot['Date'] <= pd.to_datetime('2021-12-01'))

#Fill the axis with data
axes[0].scatter(df_to_plot.Date[mask], df_to_plot.Wind[mask], s=2, label = "Wind at JFK", color = 'g')  
axes[1].scatter(df_to_plot.Date[mask], df_to_plot.Mean_delay[mask], s=2, label = "Mean delay at JFK", color = 'r') 
axes[2].scatter(df_to_plot.Date[mask], df_to_plot.Precipitation[mask], s=2, label = "Rain at JFK", color = 'b') 

#Set title and axis legend, only set axis legend on the sides
axes[0].legend(loc = 'upper left')
axes[1].legend(loc = 'upper left')
axes[2].legend(loc = 'upper left')

axes[0].set_ylabel('Average wind, m/s')
axes[1].set_ylabel('Average delay, min')
axes[2].set_ylabel('Daily precip., in')
axes[2].set_xlabel('Date')

# Remove the bounding box to make the graphs look less cluttered
#axes[0].spines['right'].set_visible(False)
#axes[0].spines['top'].set_visible(False)
#Format ticks
for tick in axes[0].get_xticklabels():
    tick.set_rotation(45)
for tick in axes[1].get_xticklabels():
    tick.set_rotation(45)
for tick in axes[2].get_xticklabels():
    tick.set_rotation(45)
plt.show()
fig.savefig(f"Delays and weather at daily levels.jpg", bbox_inches='tight', dpi = 300)

# COMMAND ----------

# MAGIC %md
# MAGIC #Make correlation matrix

# COMMAND ----------

df_to_plot['Wind_z'] = (df_to_plot['Wind'] - df_to_plot['Wind'].mean()) / df_to_plot['Wind'].std() 
df_to_plot['Precipitation_z'] = (df_to_plot['Precipitation'] - df_to_plot['Precipitation'].mean()) / df_to_plot['Precipitation'].std() 
df_to_plot['Mean_delay_z'] = (df_to_plot['Mean_delay'] - df_to_plot['Mean_delay'].mean()) / df_to_plot['Mean_delay'].std() 

# COMMAND ----------

features = ['Wind','Mean_delay','Precipitation']
fig, ax = plt.subplots()
corr_mtx = df_to_plot[features].corr()
c = ax.matshow(corr_mtx)
ax.set_xticks(np.arange(len(features)), labels=features)
ax.set_yticks(np.arange(len(features)), labels=features)
fig.colorbar(c, ax = ax) 
ax.set_title('Correlation of delays and weather')

# Loop over data dimensions and create text annotations.
for i in range(len(features)):
    for j in range(len(features)):
        text = ax.text(j, i, round(corr_mtx.iloc[i, j],2),
                       ha="center", va="center", color="w")

fig.savefig(f"Correlation of delays and weather.jpg", bbox_inches='tight', dpi = 300)
plt.show()

# COMMAND ----------

plot_acf(df_to_plot.Mean_delay, lags = 50)
plt.show()

# COMMAND ----------


