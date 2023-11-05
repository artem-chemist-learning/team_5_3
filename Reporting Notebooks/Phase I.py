# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Airline Delays: Phase 1 Proposal

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
# MAGIC |Bailey Kuehl|Week 1|
# MAGIC |Art|Week 2|
# MAGIC |Erik|Week 3|
# MAGIC |Lucy|Week 4|

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
# MAGIC
# MAGIC #### Airlines    
# MAGIC [Lucy]
# MAGIC
# MAGIC
# MAGIC #### Weather
# MAGIC
# MAGIC The weather dataset comes from NOAA (National Oceanic and Atmospheric Administration).  After review of their website, various datasets, and documentation, we found that our dataset best aligns with their Local Climatological Data (LCD) dataset.  This dataset consists of hourly, daily and monthly weather observation summaries.  Below is a table showing the general sections of features, some examples from each, and our understanding of those features.  
# MAGIC   
# MAGIC <br>
# MAGIC <html>
# MAGIC   <head>
# MAGIC     <title><b>Weather Dataset Features<b/></title>
# MAGIC   </head>
# MAGIC   <body>
# MAGIC     <div style="text-align: center;">
# MAGIC       <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/weather_features.png?raw=true" width="800">
# MAGIC    </div>
# MAGIC   </body>
# MAGIC </html>
# MAGIC <br>
# MAGIC
# MAGIC An important discovery from our review of the documentation of this dataset is how the hourly, daily, and monthly values are populated in the dataset. The LCD documenation states: "After each day’s hourly observations, the daily data for that day is given in the following row and monthly data is in the row following the final hourly observations for the last day in the month."  To better understand this we reviewed all weather observations for a specific station **Los Angeles Airport (LAX)** for a specific day **1/10/15**.  There were a total of 40 observations recorded that specific day, accross (4) different report types. Each report type had varied observation frequencies and features included. Below is a table showing those details. There was a clear delineation between the records that held hourly vs. daily observations.
# MAGIC
# MAGIC <br>
# MAGIC <html>
# MAGIC   <head>
# MAGIC     <title><b>LAX - 1/10/15: Weather Observations<b/></title>
# MAGIC   </head>
# MAGIC   <body>
# MAGIC     <div style="text-align: center;">
# MAGIC       <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/lax_weather_sample_20150110.png?raw=true" width="400">
# MAGIC     </div>
# MAGIC   </body>
# MAGIC </html>
# MAGIC <br>
# MAGIC
# MAGIC #### Join Logic
# MAGIC
# MAGIC **Location**
# MAGIC
# MAGIC After our review of the separate flight and weather datasets, we then sought to understand the logic used for the join of the combined datasets.  Every record of the joined dataset consists of flight information, as well as weather station information representative for both the arriving and departing airports.  From this knowledge, it is our general assumption that a mapping was performed to join each airport to a corresponding weather station, likely by closest proximity using longitude and latitude coordinates.
# MAGIC
# MAGIC While the weather station information is populated for both the departing and arriving airports, the actual weather observations are only provided for the departing airport. 
# MAGIC
# MAGIC **Time**
# MAGIC
# MAGIC From are review of the raw weather data, and the understanding of composition, the time-based logic used for the join needed to be understood.  We knew that each flight record was only being joined to one weather observation record.  What we sought to understand is what logic was used for time. A small sample of the time-based components of the joined dataset is presented below. Based off this sample, it appears that the "4 hours prior departure (UTC)" timestamp was used to find the next available corresponding timestamp in the weather data table. 
# MAGIC
# MAGIC <br>
# MAGIC <html>
# MAGIC   <head>
# MAGIC     <title><b>Time-Based Components</b></title>
# MAGIC   </head>
# MAGIC   <body>
# MAGIC     <div style="text-align: center;">
# MAGIC       <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/timejoin_sample.png?raw=true" width="700">
# MAGIC     </div>
# MAGIC   </body>
# MAGIC </html>
# MAGIC <br>
# MAGIC
# MAGIC Based on this logic, that means that flights may be joined to inconsistent types of weather observations, soley based on the timing of that specific flight.  Most flights will be joined to hourly records, while others may coincidentally be joined to other reported summaries (like daily summaries).  Additionally, the documentation for the weather data also stated that timestamps are in local time zones, whereas the weather data appears to be in UTC. Further exploration will be needed to better understand the join logic used and it's implications. Next we speak to summary statistics for the joined dataset. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Data
# MAGIC
# MAGIC #### Nulls
# MAGIC In the combined weather / airline 3 month dataset, there are **92 columns** that had missing (null) values for > 90% of the entries. We have chosen 90% as the maximum allowable threshold nulls and use this as a feature selection method. Below are some features which have been dropped due to exceeding the threshold:
# MAGIC
# MAGIC |Feature|% Nulls| 
# MAGIC |---|---|
# MAGIC |ShortDuration . . .| 100%|
# MAGIC |Monthly . . .| 100%|
# MAGIC |. . .|. . .|
# MAGIC |DailySnowDepth| 99.84%|
# MAGIC |DailyPrecipitation|99.77%|
# MAGIC |DailySustainedWindSpeed|99.77%|
# MAGIC
# MAGIC This leaves us with 124 remaining features to use for modeling.
# MAGIC
# MAGIC #### Statistics
# MAGIC
# MAGIC |Feature|Mean|Std Dev|Range
# MAGIC |---|---|---|---|
# MAGIC |---|---|---|---|
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dealys and weather at JFK at daily level
# MAGIC <br>
# MAGIC <html>
# MAGIC   <body>
# MAGIC     <div style="text-align: center;">
# MAGIC       <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/Delays and weather at daily levels.jpg?raw=true" width="400">
# MAGIC    </div>
# MAGIC   </body>
# MAGIC   <body>
# MAGIC     <div style="text-align: center;">
# MAGIC       <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Code/Correlation of delays and weather.jpg?raw=true" width="400">
# MAGIC    </div>
# MAGIC   </body>
# MAGIC </html>
# MAGIC <br>

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
# MAGIC A blogpost <a href="https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20">on medium</a> and <a href="https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/">this tutorial</a> have detailed outline of the process
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Success Metrics
# MAGIC To measure usefulness of the model we will use precision at 80% recall.
# MAGIC
# MAGIC Given the imbalanced nature of the dataset simple accuracy would not be the optimal metric. On such dataset there is usually little difference in accuracy between a usable model and the one that does not yield any actionable insight. Interplay between precision and recall determines usefulness of the model in this case.
# MAGIC $$Precision = \frac{\text{correctly predicted as delayed}}{\text{all flights predicted as delayed}} = \frac{\text{True Positives}}{\text{True Positives + False Positive}}$$
# MAGIC
# MAGIC $$Recall = \frac{\text{correctly predicted as delayed}}{\text{all flights actually delayed}}= \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$$
# MAGIC Normally the output of already trained model can be easily tuned to increase one metric at the expense of another. Therefore, a compound metric that combines precision and recall into one number is used to compare different models. Examples of such metrics are F1-score and ROC AUC.
# MAGIC $$F1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$$
# MAGIC These metrics allow for easy comparision between models, but do not measure model usefullness directly. For instance,  a model can have very high ROC AUC, but not demonstrate acceptabel precision at any level of recall, because of the shape of ROC curve.
# MAGIC
# MAGIC To come up with a usefull metric we have to make certain assumptions:
# MAGIC - The model will be used by airlines to better allocate resources in case of a delayed flight
# MAGIC - Currently, airlines already handle this probelm in some fasion and the current solution is already acceptable, i.e. the problem is not severe.
# MAGIC - The cost of small delay, a few minutes over 15 min cut off, is likely 0. Most passengers have a few hours between connecting flights and 15 min delay does not need any action.
# MAGIC - The cost of action in case of the delayed flight can not be 0. If airline decides to act on the model prediction, some resources will be spent on acting on this prediction. 
# MAGIC
# MAGIC Based on these assumptions we conclude that low recall is more tolerable than low precision. Lots of flights mislabeled as delayed will inevitably result in noticible cost (i.e. accumulating many flase positives becomes expensive). On the other hand, flights that model misses will likely not casue any problem (i.e. flase negatives are mostly cheap). At the same time, if the recall is very low and the model misses most delayed flights, there will be instances when the model overlooks a significant delay. Large cost will be incured by the airline and they will stop using this model after just a few of these costly mistakes (i.e. there are rare false neagtives that are expensive). Without in-depth domain knowledge we postulated that 80% recall is an accpetable compromise.
# MAGIC
# MAGIC With this acceptably low recall we will optimise our models to achive maximum precision. This way the airline can be certain that the flight flagged as delayed is actually going to be delayed and the resources expended on dealing with the delay are not spent in vain. At the same time, the airline can continue using its current practices to deal with the few short delays overlooked by the model.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gantt Chart & Credit Assignment Table
# MAGIC
# MAGIC Below is our project plan, timeline, and credit assignment table.  We have broken down the project into phases and subsequent tasks and assigned a lead for each. We have anticipated the workout and time duration for each step in order to successfully complete the project by the provided deadlines. We plan to use this to guage our pace and progress on the project and will update as the project evolves. 
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

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Pipeline
# MAGIC
# MAGIC ### 1. Data cleaning and preprocessing
# MAGIC
# MAGIC * Re-format data into correct datatypes
# MAGIC * Eliminate features with >90% null observations
# MAGIC * Encode categorical features
# MAGIC * Normalize numerical features
# MAGIC * Impute misssing data
# MAGIC
# MAGIC ### 2. Feature selection
# MAGIC
# MAGIC * Data points that have > 90% nulls will be dropped
# MAGIC * Use Logistic Regression with Lasso Regularization to select features with large weights
# MAGIC
# MAGIC ### 3. Model training
# MAGIC
# MAGIC * Train a Baseline statistical model predicting average delay time for all flights. 
# MAGIC * Train a Random Forest model to predict delays more than 15 minutes
# MAGIC
# MAGIC ### 4. Model evaluation
# MAGIC
# MAGIC * Evaluate the performance of the trained model on a holdout dataset
# MAGIC * Use precision at 80% recall to compare baseline, logistic regression and random forest.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Algorithms
# MAGIC Baseline algorim will be based on the delays at the airport of departure observed over last four hours. If the airport had less than 3 departueres in that timewindow, the departures from all airports located in the same state will be used. This algorithm requires no machine learning at all, yet allows us to implement our success metric. Given the set of delays that occured over the last 4 hours, we can hypothesize that mean of that distribution is below 15 min. If a one-tail hypothesis test rejects this hypothesis, we predict a delay. Adjusting confidence level of the test we can control precision and recall of this model. Once we find confidence level such that recal is >= 80% we will compute precision of this model and use it as a benchmark.
# MAGIC
# MAGIC Logistic regression (LR) will be the first machine learning algorithm that we will use. We will train the model using traditional Binary Cross Entropy as loss function. We also consider using lasso regularization to combat overfitting. Using L1 as opposed to L2 at this stage will allow us to use LR as a feature selector for more sophisticated models. Initially, we will train on all avaliable features and expect to have most of them to have very low weights. In the  future exploration we will use features with weights above certian cut-off, that we will define later. To calcualte the sucess metric we will adjust the decision cut off so that the recall is ~80% and then use precision at this cut-off to compare LR performance to baseline model.
# MAGIC
# MAGIC We also plan to explore more sophisticated algorithms such as Random Forest, possibly with gradient boosting. We will use features identified as significant using Logistic funcation. An issue with RF is that its precision/recall balance is adjusted using hyper-parameters used in model training and computationally expensive re-training is needed to find parameters that correspond to 80% recall. Therefore we will use 3-month subsection of dataset to adjust number of trees in the foreset, max number of levels in each decision tree and other hyperparameters to achieve ~ 80% recall. <a href="https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65">This post</a> gives a conceptual overview of the process. We will then use thse hyperparameters to train on the full dataset and hope that the resulting recall will be close to 80%. We will use precision achieved with these hyperprameters to compare to the sucess metrics of the baseline and LR models.
# MAGIC
# MAGIC As a potential alternative we will consider using Facebook prophet, which is specifically deigned to deal with timeseries with pronounced seasonality and has a potential to outperform other models. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and Next Steps
# MAGIC [Bailey]
# MAGIC
# MAGIC #### Challenges
# MAGIC - Seasonality of data
# MAGIC - COVID period at the end of data
# MAGIC - Efficiency of operations on large data
# MAGIC - Missing data
# MAGIC
# MAGIC #### Conclusions 
# MAGIC - TBD: We plan to draw conclusions from our data in Phases 2 and 3.
# MAGIC
# MAGIC #### Next Steps
# MAGIC - In depth ETA on joined dataset
# MAGIC - Feature selection and engineering
# MAGIC - Generating a baseline
# MAGIC
