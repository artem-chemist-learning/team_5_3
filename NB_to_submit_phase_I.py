# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # We gonna make it

# COMMAND ----------

# MAGIC %md
# MAGIC ##The Team
# MAGIC <pre>     Artem Lebedev          Lucy Moffitt Herr          Erik Sambrailo           Bailey Kuehl<pre>
# MAGIC  artem.lebedev@berkeley.edu       lherr@berkeley.edu          e.sambrail0@berkeley.edu      bkuehl@berkeley.edu
# MAGIC <div>
# MAGIC <img src="files/tables/art.png" width="200"/> <img src="files/tables/lucy.png" width="200"/> <img src="files/tables/erik.png" width="200"/> <img src="files/tables/bailey.png" width="200"/>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Phase leader plan
# MAGIC Each person assumes the project manager for one week (4 work weeks for the project)
# MAGIC
# MAGIC |Person|Week|
# MAGIC |---|---|
# MAGIC |Art|Week 1|
# MAGIC |Art|Week 2|
# MAGIC |Art|Week 3|
# MAGIC |Art|Week 4|

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Credit assignment plan 
# MAGIC Set up the tasks to be completed by phase; estimate the effort in terms of person-hours to complete the task, and assign the person responsible for that task
# MAGIC
# MAGIC |Task|Phase|Person|Hours
# MAGIC |---|---|---|---|
# MAGIC |Blob setup|1|Art||
# MAGIC |Select Eval Metrics|1|Art||
# MAGIC |Gantt Chart|1|Erik||
# MAGIC |Explain Data - Weather|1|Erik||
# MAGIC |Explain Data - Airlines|1|Lucy||
# MAGIC |Algorithm|1|Lucy, Erik||
# MAGIC |Train, test split|1|Bailey|1.5|
# MAGIC |Abstract|1|Bailey|0.5|
# MAGIC |Conclusions, challenges, next steps|1|Bailey|0.5|
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Abstract
# MAGIC [Bailey]
# MAGIC
# MAGIC Project abstract in ~150 words (remember to use the STAR methodology, we will not count words)
# MAGIC In your own words, summarize at a high level the project, the data, your planned pipelines/experiments, metrics, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data description
# MAGIC Data you plan to use
# MAGIC
# MAGIC Basic analysis and understanding of data
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary visual EDA
# MAGIC
# MAGIC Data you plan to use
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Algorithms to be explored
# MAGIC Which machine learning algorithm(s) are you considering using and why?
# MAGIC
# MAGIC Description of algorithms to be used (list the names, implementations, loss functions that you will use)
# MAGIC Description of metrics and analysis to be used (be sure to include the equations for the metrics)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Test Split
# MAGIC [Bailey]
# MAGIC
# MAGIC We cannot choose random samples for our test, train, and validation datasets due to the nature of time series data. We also want to avoid future-looking when we train our model.
# MAGIC
# MAGIC Instead, we use cross-validation on a rolling basis to split our data into test and train. This process involves starting with a small subset of data for training purposes, forecasting the later data points, followed by checking the accuracy of the forecasted data points. These forecasted data points are then included as part of the next training dataset and subsequent data points are forecasted.
# MAGIC
# MAGIC Thus, we divide the training set into two folds at each iteration such that the validation set is always ahead of the training set. 
# MAGIC
# MAGIC Resources:    
# MAGIC https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20
# MAGIC https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metrics of success
# MAGIC Metrics that you might use to measure success (standard metrics and domain-specific metrics)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gant chart
# MAGIC A block diagram (Gantt diagram) of the key steps involved in completing this task and a timeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline
# MAGIC Description of the pipeline steps you plan to use (and a block diagram)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and Next Steps
# MAGIC [Bailey]
# MAGIC
# MAGIC ### Challenges
# MAGIC - Seasonality of data
# MAGIC - Efficiency of operations on large data
# MAGIC - Missing data
# MAGIC
# MAGIC ### Conclusions 
# MAGIC - TBD
# MAGIC
# MAGIC ### Next Steps
# MAGIC - In depth ETA on joined dataset
# MAGIC - Feature selection and engineering
# MAGIC - Generating a baseline
# MAGIC
