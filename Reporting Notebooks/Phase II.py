# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Airline Delays: Phase 2 Progress

# COMMAND ----------

# MAGIC %md
# MAGIC ##The Team
# MAGIC <pre>     Artem Lebedev          Lucy Moffitt Herr          Erik Sambrailo           Bailey Kuehl<pre>
# MAGIC  artem.lebedev@berkeley.edu       lherr@berkeley.edu          e.sambrail0@berkeley.edu      bkuehl@berkeley.edu
# MAGIC <div>
# MAGIC <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/art.png?raw=true" width="200"> <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/lucy.png?raw=true" width="200"> <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/erik.png?raw=true" width="200"> <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/bailey.png?raw=true" width="200">
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
# MAGIC |Bailey|Week 1|
# MAGIC |**Art**|**Week 2**|
# MAGIC |**Erik**|**Week 3**|
# MAGIC |Lucy|Week 4|

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Abstract
# MAGIC ### Phase II Abstract
# MAGIC 1. Context of the project
# MAGIC 2. Methods used
# MAGIC 3. Findings/results
# MAGIC 4. Next steps
# MAGIC
# MAGIC ### Phase I Abstract
# MAGIC Air travel in the United States is the preferred method of transportation for commercial and recreational use because of its speed, comfort, and safety [1]. Given its initial popularity, air travel technology has improved significantly since the first flight took off in 1908 [2]. For example, modern forecasting technology allows pilots to predict the optimal route and potential flight delays and cancellations given forecasted headwinds, storms, or other semi-predictable events. However, previous studies have found that weather is actually not the primary indicator of whether a flight will be delayed or canceled [1]. Today, seemingly unexpected flight delays are not only a nuisance for passengers, but also could a potentially detrimental threat to the airline industry if customers continue to lose trust in public airline capabilities. Thus, the primary goal of this project is to predict flights delays more than 15 minutes in duration that occur within 2 hours prior to the expected departure time. To accomplish this, we will extract airline and weather data spanning the years 2015 - 2019 from the *On Time Performance and Weather (OTPW)* dataset [3]. Feature selection will be performed through null thresholding (dropping features with more than 90% nulls) and lasso regularization. Key features are hypothesized to be Airline (e.g. *Delta, Southwest*), expected maintenence, history of delays for a given flight number (route), and extreme weather (e.g. ice or snow) [4]. We will perform data cleaning, imputation, and exploratory analysis on the remaining data. The cleaned data will be split into test, train, and validation sets via cross-validation on a rolling basis given the time series nature of the data. We will then build and train a logisitic regression model as a baseline, as well as a random forest to predict delays. The proposed pipeline is expected to perform efficiently in terms of runtime given our proposed utilization of partitionable parquet files in place of the more commonly used CSV files. Finally, to measure the success of our model, we propose to use precision and recall, optimizing the tradeoff between the two such that precision is maximized given a goal recall of 80%. Using the results from this project, we hope to advise airlines on key factors affecting flight delays so that they can mitigate them to better satisfy their customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Remaning Data
# MAGIC
# MAGIC Description of the data and task at hand
# MAGIC
# MAGIC --Data description
# MAGIC
# MAGIC --Task to be tackled
# MAGIC
# MAGIC -- Provide diagrams to aid understanding the workflow

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Modeling Pipeline
# MAGIC Below is a written explanation of our proposed pipeline, as well as a block diagram to demonstrate the workflow.
# MAGIC
# MAGIC #### 0. Data Ingestion
# MAGIC * Ingest CSV files and represent the data as PySpark Dataframes
# MAGIC * Checkpoint data as Parquet file
# MAGIC   * Partition data on an hourly basis using Structured Streaming [4]
# MAGIC
# MAGIC #### 1. Data cleaning and preprocessing
# MAGIC
# MAGIC * Reformat data into correct datatypes
# MAGIC * Eliminate features with >90% null observations
# MAGIC * Encode categorical features
# MAGIC * Normalize numerical features
# MAGIC * Impute missing data
# MAGIC
# MAGIC #### 2. Feature selection
# MAGIC
# MAGIC * Data points that have > 90% nulls will be dropped
# MAGIC * Use Logistic Regression with Lasso Regularization to select features with large weights
# MAGIC
# MAGIC #### 3. Model training
# MAGIC
# MAGIC * Train a Baseline statistical model predicting average delay time for all flights. 
# MAGIC * Train a Random Forest model to predict delays more than 15 minutes
# MAGIC
# MAGIC #### 4. Model evaluation
# MAGIC
# MAGIC * Evaluate the performance of the trained model on a holdout dataset
# MAGIC * Use precision at 80% recall to compare baseline, logistic regression and random forest.
# MAGIC * Perform additional model tuning and feature engineering
# MAGIC
# MAGIC
# MAGIC <br>
# MAGIC <html>
# MAGIC   <head>
# MAGIC     <title><b>Pipeline Block Diagram</b></title>
# MAGIC   </head>
# MAGIC   <body>
# MAGIC     <div style="text-align: center;">
# MAGIC       <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/phase1_pipeline_model_v2.png?raw=true" width="900">
# MAGIC     </div>
# MAGIC   </body>
# MAGIC </html>
# MAGIC <br>
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC
# MAGIC #### Summary
# MAGIC
# MAGIC #### Discussion
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and Next Steps
# MAGIC #### Conclusions 
# MAGIC
# MAGIC
# MAGIC #### Next Steps
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## References
# MAGIC Inline citations throughout the report are represented as bracketed references, e.g. *[4]*.     
# MAGIC <br>
# MAGIC
# MAGIC 1. Analysis of the Influence of Factors on Flight Delays in the United States Using the Construction of a Mathematical Model and Regression Analysis: https://ieeexplore.ieee.org/document/9795721
# MAGIC 2. The world’s first successful controlled powered flight: https://nzhistory.govt.nz/media/video/wright-brothers-first-flight-1903#:~:text=In%201903%2C%20Americans%20Orville%20and,Wright%20Brothers%20flying%20in%201908. 
# MAGIC 3. On Time Performance and Weather (OTPW) Dataset, original source: https://www.transtats.bts.gov/homepage.asp 
# MAGIC 4. Prediction of weather-induced airline delays based on machine learning algorithms: https://ieeexplore.ieee.org/document/7777956
# MAGIC 5. Parquet files: https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html
# MAGIC 6. Time Series and Cross Validaition: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20
# MAGIC 7. Code: https://chat.openai.com/
# MAGIC 8. Shrivastava, S. (2020). "Cross Validation in Time Series." Medium. https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20
# MAGIC 9. "Cross-Validation strategies for Time Series forecasting" [Tutorial]. Packt. https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
# MAGIC

# COMMAND ----------

# DBTITLE 1,Appendix
# MAGIC %md
# MAGIC ## Gantt Chart & Credit Assignment Table
# MAGIC
# MAGIC Below is our project plan, timeline, and credit assignment table.  We have broken down the project into phases and subsequent tasks and assigned a lead for each. We have anticipated the workout and time duration for each step in order to successfully complete the project by the provided deadlines. We plan to use this to gauge our pace and progress on the project and will update as the project evolves. 
# MAGIC
# MAGIC <br>
# MAGIC <html>
# MAGIC   <head>
# MAGIC     <title><b>Project Plan & Timeline<b/></title>
# MAGIC   </head>
# MAGIC   <body>
# MAGIC     <div style="text-align: center;">
# MAGIC       <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/Gant_%26_Credit_Plan.png?raw=true" width="1600">
# MAGIC    </div>
# MAGIC   </body>
# MAGIC </html>
# MAGIC <br>
