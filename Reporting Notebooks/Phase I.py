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
# MAGIC |Bailey|Week 1|
# MAGIC |Art|Week 2|
# MAGIC |Erik|Week 3|
# MAGIC |Lucy|Week 4|

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Abstract
# MAGIC
# MAGIC Air travel in the United States is the preferred method of transportation for commercial and recreational use because of its speed, comfort, and safety [1]. Given its initial popularity, air travel technology has improved significantly since the first flight took off in 1908 [2]. For example, modern forecasting technology allows pilots to predict the optimal route and potential flight delays and cancellations given forecasted headwinds, storms, or other semi-predictable events. However, previous studies have found that weather is actually not the primary indicator of whether a flight will be delayed or canceled [1]. Today, seemingly unexpected flight delays are not only a nuisance for passengers, but also could a potentially detrimental threat to the airline industry if customers continue to lose trust in public airline capabilities. Thus, the primary goal of this project is to predict flights delays more than 15 minutes in duration that occur within 2 hours prior to the expected departure time. To accomplish this, we will extract airline and weather data spanning the years 2015 - 2019 from the *On Time Performance and Weather (OTPW)* dataset [3]. Feature selection will be performed through null thresholding (dropping features with more than 90% nulls) and lasso regularization. Key features are hypothesized to be Airline (e.g. *Delta, Southwest*), expected maintenence, history of delays for a given flight number (route), and extreme weather (e.g. ice or snow) [4]. We will perform data cleaning, imputation, and exploratory analysis on the remaining data. The cleaned data will be split into test, train, and validation sets via cross-validation on a rolling basis given the time series nature of the data. We will then build and train a logisitic regression model as a baseline, as well as a random forest to predict delays. The proposed pipeline is expected to perform efficiently in terms of runtime given our proposed utilization of partitionable parquet files in place of the more commonly used CSV files. Finally, to measure the success of our model, we propose to use precision and recall, optimizing the tradeoff between the two such that precision is maximized given a goal recall of 80%. Using the results from this project, we hope to advise airlines on key factors affecting flight delays so that they can mitigate them to better satisfy their customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data description
# MAGIC
# MAGIC #### Airlines    
# MAGIC The flights dataset is sourced from the TranStats data collection (U.S. Department of Transportation). The full dataset consists of on-time performance data for 31,746,841 U.S. passenger flights between 2015 and 2021, with 109 total features.
# MAGIC
# MAGIC Key features to use in our EDA and modeling include flight and carrier identifiers, airport location information, and delay time and source attributes.
# MAGIC
# MAGIC Unique identifiers for flights include
# MAGIC - 'Reporting_Airline': airline/carrier (2-character code with optional numeric suffix linking multiple carriers)
# MAGIC - 'TailNumber': plane (2-6 character alphanumeric code)
# MAGIC - 'Flight_Number_Reporting_Airline': flight number (2-character airline + 1-4 digit number)
# MAGIC
# MAGIC Airport location features include
# MAGIC - 'OriginAirportID' and 'DestAirportID': unique airport ID (numeric 5-digit code). This identifier is consistent over multiple years. 
# MAGIC - 'Origin' and 'Dest': airport name. These strings can be parsed to extract airport type (regional vs. international). 
# MAGIC - ‘OriginCity’ and ’DestCity’: airport city name
# MAGIC - 'OriginState' and 'DestState': airport state (2-character code, e.g., "AL" for Alaska)
# MAGIC - ‘Distance’: distance between airports in miles 
# MAGIC - ‘Flights’: total number of flights 
# MAGIC
# MAGIC Features describing the timing and source of delays include 
# MAGIC - 'DepDelay' and 'ArrDelay': difference in minutes between scheduled and actual times of departure and arrival. Negative values represent early flights. 
# MAGIC - 'TaxiOut' and 'TaxiIn': on-runway time in minutes at departure and arrival
# MAGIC - 'WheelsOff' and ‘WheelsOn’: local time (in hh:mm) of takeoff and landing
# MAGIC - 'CarrierDelay', ‘WeatherDelay',’NASDelay',’SecurityDelay', ‘LateAircraftDelay’: delay time in minutes based on attributed source. Note: these features contain ~80% null values in our 2015 subset but may provide some insight in EDA. 
# MAGIC - 'Cancelled' and 'Diverted': indicator variables representing flight cancellations or diversions to other airports
# MAGIC
# MAGIC Lastly, numeric time period features ('DayofMonth', 'Month', 'Year') will be critical to examining patterns of delays over time, performing the train-test split and time-series cross validation, and facilitating joins with the weather data. 
# MAGIC <br>
# MAGIC <br>
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
# MAGIC From our review of the raw weather data, and the understanding of composition, the time-based logic used for the join needed to be understood.  We knew that each flight record was only being joined to one weather observation record.  What we sought to understand is what logic was used for time. A small sample of the time-based components of the joined dataset is presented below. Based off this sample, it appears that the "4 hours prior departure (UTC)" timestamp was used to find the next available corresponding timestamp in the weather data table. 
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
# MAGIC ## Numerical Summary of Data
# MAGIC
# MAGIC #### Nulls
# MAGIC In the combined OTPW 3-month dataset (2015), there are **92 columns** that had missing (null) values for > 90% of the entries. We have chosen 90% as the maximum allowable threshold nulls and use this as a feature selection method. Below are some features which have been dropped due to exceeding the threshold. *(Note:  "ShortDuration . . ." notation is used to represent all columns that begin with "ShortDuration" as all of these columns have been dropped. We use the same notation for columns starting with "Monthly".)*
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
# MAGIC Removing these columns leaves us with 124 remaining features to use for modeling. On the remaining columns, we have identified some of the important numerical columns and find the following summary statistics for them:
# MAGIC
# MAGIC #### Statistics
# MAGIC
# MAGIC |Feature|Mean|Std Dev|Range|
# MAGIC |---|---|---|---|
# MAGIC |Dep_Delay (min)|10.357|37.846|(-1.0 - 996.0)|
# MAGIC |Arr_Delay (min)|6.238|40.515|(-1.0 - 998.0)|
# MAGIC |Taxi_Out (min)|16.396|9.630|(1.0 - 99.0)|
# MAGIC |Taxi_In (min)|7.462|6.103|(1.0 - 99.0)|
# MAGIC |Carrier_Delay (min)|18.280|46.303|(0.0 - 996.0)|
# MAGIC |HourlyDewPointTemp (degrees F)|32.179|19.185|(0.0 - 9.0)|
# MAGIC |HourlyVisibility (miles) |9.008|2.406|(0.0 - 9.94)|
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization: Delays and weather at JFK at daily level
# MAGIC To take a closer look at some of the flight delay data, we provide some visuals for JFK Airport - one of the largest, busiest airports in the United States.
# MAGIC
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
# MAGIC ## Train / Test Split
# MAGIC
# MAGIC We cannot choose random samples for our test, train, and validation datasets due to the nature of time series data, as this could cause data-leakage and future-looking (i.e. inferring the trend of future samples).
# MAGIC
# MAGIC Instead, we will use cross-validation on a rolling basis to split our data into test and train. This process involves starting with a small subset of data for training purposes and forecasting the later data points, followed by checking the accuracy of the forecasted data points. These forecasted data points are then included as part of the next training dataset and subsequent data points are forecasted [5, 8, 9]. Thus, we divide the training set into two folds at each iteration such that the validation set is always ahead of the training set.
# MAGIC
# MAGIC We have chosen a 60 / 20 / 20 percentage split for train, test, and validation, respectively.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Success Metrics
# MAGIC To measure the performance of the model, we will use precision at 80% recall.
# MAGIC
# MAGIC Given the imbalanced nature of this dataset, simple accuracy would not be the optimal metric. On such data, there is usually little difference in accuracy between a usable model and the one that does not yield any actionable insight. Interplay between precision and recall determines usefulness of the model in this case.
# MAGIC $$Precision = \frac{\text{correctly predicted as delayed}}{\text{all flights predicted as delayed}} = \frac{\text{True Positives}}{\text{True Positives + False Positive}}$$
# MAGIC
# MAGIC $$Recall = \frac{\text{correctly predicted as delayed}}{\text{all flights actually delayed}}= \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$$
# MAGIC Normally, the output of trained model can be easily tuned to increase one metric at the expense of another. Therefore, a compound metric that combines precision and recall into one metric can be used to compare different models. Examples of such metrics are F1-score and ROC AUC.
# MAGIC $$F1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$$
# MAGIC These metrics allow for easy comparision between models, but do not measure model usefulness directly. For instance, a model can have very high ROC AUC but not demonstrate acceptable precision at any level of recall because of the shape of ROC curve.
# MAGIC
# MAGIC To come up with a useful metric, we have to make certain assumptions:
# MAGIC - The model will be used by airlines to better allocate resources in case of a delayed flight
# MAGIC - The cost of small delay, a few minutes over 15 min cut off, is likely 0. Most passengers have sufficient time between connecting flight to incur 15-minute delays without substantial cost or inconvenience.
# MAGIC - The cost of action in case of the delayed flight is non-zero.
# MAGIC
# MAGIC Based on these assumptions, we conclude that low recall is a more tolerable success metric for this context than low precision. Many on-time flights misclassified as delayed will inevitably result in noticible cost (i.e. accumulating many false positives becomes expensive). On the other hand, actually delayed flights that the model misclassifies as on-time as will likely cause fewer problems (i.e. false negatives are less expensive). At the same time, if the recall is very low and the model misses most delayed flights, there will be instances when the model overlooks a significant delay. Large cost will be incured by the airline, and they may stop using this model after just a few of these costly mistakes (i.e. there are rare false neagtives that are expensive). Without in-depth domain knowledge, we postulate that 80% recall is an acceptable compromise.
# MAGIC
# MAGIC With this acceptably low recall, we will optimize our models to achieve maximum precision. This way, the airline can be certain that a flight flagged as delayed is actually going to be delayed and the resources expended on addressing the delay are not spent in vain. At the same time, the airline can continue using its current practices to deal with the few shorter-duration delays overlooked by the model.
# MAGIC

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Pipeline
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
# MAGIC ## Algorithms
# MAGIC Our baseline algorithm will be based on the delays at the airport of departure observed over the previous four hours. If the airport had less than 3 departures in that time window, the departures from all airports located in the same state will be used. This algorithm requires no machine learning at all, yet allows us to implement our success metric. Given the set of delays that occured over the last 4 hours, we can hypothesize that mean of that distribution is below 15 minutes. If a one-tail hypothesis test rejects this hypothesis, we predict a delay. Adjusting the confidence level of the test enables us to control the precision and recall of this model. Once we reach a confidence level such that recall is >= 80%, we will compute the precision of the model and use it as a benchmark.
# MAGIC
# MAGIC Logistic regression (LR) will be the first machine learning algorithm that we will use. We will train this model using traditional Binary Cross Entropy as its loss function. We will also consider using lasso regularization to combat overfitting. Using L1 as opposed to L2 at this stage will allow us to use LR as a feature selector for more sophisticated models. Initially, we will train on all available features with the expectation that most will have very low weights. In subsequent exploration, we will use features with weights above the determined cut-off that we ultimatley define. To calculate the sucess metric, we will adjust the decision cut-off to obtain a recall value of ~80%, then use the precision at this cut-off to compare LR performance to the baseline model.
# MAGIC
# MAGIC We also plan to explore more sophisticated algorithms such as Random Forest, possibly with gradient boosting. We will use features identified as significant using the logistic function. One anticipated issue with RF is that its precision/recall balance is adjusted using hyper-parameters used in model training, and computationally expensive re-training is needed to find parameters that correspond to 80% recall. Therefore, we will use a 3-month subset of the data to adjust number of trees in the forest, the maximum number of levels in each decision tree, and additional hyperparameters to achieve ~80% recall. <a href="https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65">This post</a> gives a conceptual overview of this process. We will then use thse hyperparameters to train on the full dataset and hope that the resulting recall will be close to 80%. We will use precision achieved with these hyperprameters to compare to the sucess metrics of the baseline and LR models.
# MAGIC
# MAGIC As a potential alternative, we will consider using Facebook Prophet, which is specifically designed for timeseries data with pronounced seasonality and has the potential to outperform other models. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and Next Steps
# MAGIC
# MAGIC #### Challenges
# MAGIC - Seasonality of data
# MAGIC - COVID period at the end of data won't be representative of future predictions
# MAGIC - Efficiency of operations on large data
# MAGIC - Missing key weather data
# MAGIC
# MAGIC #### Conclusions 
# MAGIC - TBD: We plan to draw conclusions from our data in Phases 2 and 3.
# MAGIC
# MAGIC #### Next Steps
# MAGIC - In depth ETA on joined dataset
# MAGIC - Feature selection and engineering
# MAGIC - Generating a baseline
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
