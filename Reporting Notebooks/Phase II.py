# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Airline Delays: Phase 2

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
# MAGIC Our project aims to leverage machine learning techniques to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. The goal of this work is to advise airlines on the key factors that influence delays so that they can better anticipate potential delays. Similarly to Phase 1, we utilized approximately 75% of the data in the 1-year dataset from the *On Time Performance and Weather (OTPW)* dataset -- an airline and weather dataset containing data from 2015 - 2019 [1]. During Phase 2 of our project, we sought to narrow down our dataset to ke predictive features through extensive exploratory data analysis and feature engineering. Additionally, we aimed to train a baseline model on our data by predicting the average delay, which we believe to be a good baseline because it will potentially be within the ballpark of a predicted delay, but has much room for improvement through more advanced models. Our baseline resulted in a precision and recall of **[......insert numbers......]**, respectively. Our final goal of this phase was to train a random forest classifier, chosen for its ability to specify feature importance, which resulted in a **{x % and y %}** improvement over the baseline in precision and recall, respectively. Thus, our random forest classifier was the best pipeline during this phase. This result was expected because random forest is known to reduce overfitting, have less sensitivity to outliers, and optimizes for feature importance. For the final phase of this project, we hope to iterate on our current models to further improve precision and accuracy. Such iterations might include additional feature engineering (through joining an additional dataset or generating more new features with the current data), fine-tuning model parameters, and transforming available features. We hope to optimize our model in order to gain insights about key factors affecting airline delays so that we can share our results with major airline carriers.
# MAGIC
# MAGIC ### Phase I Abstract
# MAGIC Air travel in the United States is the preferred method of transportation for commercial and recreational use because of its speed, comfort, and safety [1]. Given its initial popularity, air travel technology has improved significantly since the first flight took off in 1908 [2]. For example, modern forecasting technology allows pilots to predict the optimal route and potential flight delays and cancellations given forecasted headwinds, storms, or other semi-predictable events. However, previous studies have found that weather is actually not the primary indicator of whether a flight will be delayed or canceled [1]. Today, seemingly unexpected flight delays are not only a nuisance for passengers, but also could a potentially detrimental threat to the airline industry if customers continue to lose trust in public airline capabilities. Thus, the primary goal of this project is to predict flights delays more than 15 minutes in duration that occur within 2 hours prior to the expected departure time. To accomplish this, we will extract airline and weather data spanning the years 2015 - 2019 from the *On Time Performance and Weather (OTPW)* dataset [3]. Feature selection will be performed through null thresholding (dropping features with more than 90% nulls) and lasso regularization. Key features are hypothesized to be Airline (e.g. *Delta, Southwest*), expected maintenence, history of delays for a given flight number (route), and extreme weather (e.g. ice or snow) [4]. We will perform data cleaning, imputation, and exploratory analysis on the remaining data. The cleaned data will be split into test, train, and validation sets via cross-validation on a rolling basis given the time series nature of the data. We will then build and train a logisitic regression model as a baseline, as well as a random forest to predict delays. The proposed pipeline is expected to perform efficiently in terms of runtime given our proposed utilization of partitionable parquet files in place of the more commonly used CSV files. Finally, to measure the success of our model, we propose to use precision and recall, optimizing the tradeoff between the two such that precision is maximized given a goal recall of 80%. Using the results from this project, we hope to advise airlines on key factors affecting flight delays so that they can mitigate them to better satisfy their customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC
# MAGIC Link to EDA code: 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA Workflow Diagram
# MAGIC **** brief written description of workflow and tasks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Description
# MAGIC *** written summary of the data
# MAGIC
# MAGIC #### Raw Features Data Dictionary
# MAGIC |Feature|Description|Data Type|
# MAGIC |---|---|---|
# MAGIC ||||
# MAGIC ||||
# MAGIC ||||
# MAGIC
# MAGIC #### Dataset Size
# MAGIC |Dataset|Rows|Columns|
# MAGIC |---|---|---|
# MAGIC |Train|||
# MAGIC |Test|||
# MAGIC |Validation|||

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Modeling Pipeline {!!!!! needs update from phase I}
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
# MAGIC ### Baseline Models: Radnom Guess and Arithmetic Mean
# MAGIC These two models place performance of more sophisticated algorithms in context.
# MAGIC
# MAGIC Random guess, the absolutely simplest model, yields approximately 18% precision at any level of recall. This is becasue the dataset is heavily biased to the on-time flights, and only ~18% of the flights are delayed. The probaility of the guess controld the recall: if we label the flight delayed with 50% probability, we mislabel 50% of the delayed flights.
# MAGIC
# MAGIC A slightly more complex model only requires a glance at the departure table of the airport 2 hours prior to deprture. For this model we calculate the average departure delay at the origin, and assume that the flight in question will have the same delay. We can label the flight as "Predicted Delayed" if the predicted delay crosses a certain cut-off. At cut_off of ~2 min we achived 80% recall and ~25% precision. in other owrds, if the average delay at the airport, based on the data avaliable 2 hours ahead of departure, exceeds 2 min, then we call the flight delayed. This gave only is a slight improvement over random guess at the lower recalls, but demonstrated a huge improvement at the low recall levels (likely more relevant for passengers that for airlines)
# MAGIC
# MAGIC Link to Baseline code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1198761045465243

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trivial Logistic Regression: Only features avaliable in the original dataset, 1 Year data 
# MAGIC
# MAGIC Link to Baseline code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1871383191021758
# MAGIC
# MAGIC This model was meant to demonstrate the potential of the simplest Machine Learning algorythm with no additional input from a Data Scientist. We also used this model to do preliminary featyre selection for further exploration in subsequenst models. All avaliable features were used in the model, with categorical features encoded as dense verctors. In order to select features, we used Lasso regualrization with regularization parameter 0.001. At this point, the following features were selected:
# MAGIC  Feature|Weight
# MAGIC  --|--
# MAGIC  Is it Thursday?|0.026573215014645938
# MAGIC  Airline UA?|0.00718926061633817
# MAGIC  Airline B6?| 0.014260556036217646
# MAGIC  Airline NK?| 0.02855810219583773
# MAGIC  Is it August?| 0.041008290359131644
# MAGIC  Origin Daily Precipitation| 0.001764759214029755
# MAGIC  Origin Daily Average RelativeHumidity| 0.028534395277081803
# MAGIC  Origin Daily Sustained Wind Speed| 0.019172550243032944
# MAGIC  Origin Daily Average Relative Humidity| 0.0047175233264285755
# MAGIC  Origin Hourly Dry Bulb Temperature| 0.2105740990135358
# MAGIC  Origin Hourly Precipitation| 0.0376938092694382
# MAGIC  Dest Hourly Dry Bulb Temperature| 0.1524622069178246
# MAGIC  Dest Hourly Precipitation| 0.010683693538168454

# COMMAND ----------

# MAGIC %md
# MAGIC ### Engineered Logistic Regression: Added features suggested by the team, 1 Year data 
# MAGIC
# MAGIC Link to Baseline code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/2798664673127070
# MAGIC
# MAGIC This model was meant to test if the rational input from a Data Scientist can improve model perfomance. All features selected by the trivial LR model were used as an input, and the following features added as well:
# MAGIC - Averages for precipitation, wind gust speed, air temprature, visibility, pressure change and pressure. Averages over 3, 6 and 12 hour tiem windows were used. All time windows had 2 hour lag behind the departure time. Data for both origin and destination was used.
# MAGIC - Average flight delay at the origin, overaged over 4 hour window, lagged 2 hour prior to departure.
# MAGIC - Average flight delay for this carrier, overaged over 4 hour window, lagged 2 hour prior to departure.
# MAGIC - Departure delay of the aircarft, latest known 2 hours before departure.
# MAGIC - Number of flights scheduled to depart with 2 hours from the flight in question.
# MAGIC - Squared hourly precipitation
# MAGIC - Squared yesterday snowfal
# MAGIC
# MAGIC Lasso regularization was applied again. The table below shows the features that were found significant
# MAGIC  Feature|Weight
# MAGIC  --|--
# MAGIC  Origin Yesterday Snowfall| 0.01304660544810334
# MAGIC  Origin Precipitation over 6H window| 0.013532988320670998
# MAGIC  Origin Precipitation over 12H window| 0.0033698415281565174
# MAGIC  Average delay at the origin| 0.1986512892706633
# MAGIC  Previous delay of the aircraft| 0.47028030810554966
# MAGIC  Average delay of the carrier| 0.22544780127460018
# MAGIC  Number of flights from the origin| 0.14840908175418083

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Logistic Regression 4 year data 
# MAGIC
# MAGIC Link to Baseline code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1537154891793919
# MAGIC
# MAGIC This model was meant to test the computational performance of our approach over the large dataset. The following features ended up being significant
# MAGIC  Feature|Weight
# MAGIC  --|--
# MAGIC  Origin Precipitation over 12H window| 0.0018616954402507725
# MAGIC  Average delay at the origin| 0.30613878147981316
# MAGIC  Previous delay of the aircraft| 0.49496311984696173
# MAGIC  Average delay of the carrier| 0.34518882390187083
# MAGIC  Number of flights from the origin| 0.1776571944179384

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparative performance of LR models
# MAGIC
# MAGIC Performance metric: Precision at 80% recall
# MAGIC  Model|Training time, min | Dataset | Test performance, % | Train Perfomance, %
# MAGIC  --|--|--|--|--
# MAGIC  Random Guess| 0| 1 Year |19.7|-
# MAGIC  Baseline| 0 | 1 Year |24.8|-
# MAGIC  Trivial LR| 14 |1 Year |22.7|as
# MAGIC  Engineered LR| 23| 1 Year |30.9|as
# MAGIC  Final LR| 18 | 4 Years |as|as
# MAGIC
# MAGIC
# MAGIC All models were trained using overlapping blocks for cross validation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 1: Random Forest
# MAGIC
# MAGIC Link to RF code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/3865911540703714
# MAGIC
# MAGIC The data was split using cross validation [2].
# MAGIC
# MAGIC - insert chart for training
# MAGIC - table of features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC
# MAGIC #### Summary
# MAGIC Overall, our proposed models .........
# MAGIC
# MAGIC |Model|Precision|Recall|F-Beta|
# MAGIC |---|---|---|---|
# MAGIC |Baseline| | | |
# MAGIC |Random Forest| | | |
# MAGIC |Logistic Regression| | | |
# MAGIC
# MAGIC #### Discussion
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and Final Thoughts
# MAGIC The purpose of this project is to use machine learning to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. This work has important implications for the general public because flight delays are extremeley disruptive for both commercial and leisure passengers, flight crew, and other airline employees. The future of the airline industry is dependent on its customers, who desire safe, seamless flight experiences. We proposed that deploying machine learning modeling techniques, including logistic regression and random forest, would provide a precise, reliable methodology for predicting flight delays. To account for factors outside of those provided in our dataset, we contributed additional features through careful engineering, including **[....insert....]**. Using these features, we were able to achieve scores of **[....insert....]** for precision and recall, respectively. 
# MAGIC
# MAGIC While this result is promising, we hope to further improve our model by **[....finish thought....]**. For the final stage of our project, we hope to iterate on our current random forest model by testing different parameters and potentially adding additional relevant features to our model. One challenge we anticipate with this is attempting to engineer features through obtaining additional data: it may be difficult to find data compatible with our current dataset or to generate desired features (such as a proxiy for the current economic status of each airline carrier). We also anticipate difficulty in the time required to utilize the full dataset on our model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## References
# MAGIC Inline citations throughout the report are represented as bracketed references, e.g. *[4]*.     
# MAGIC <br>
# MAGIC
# MAGIC 1. On Time Performance and Weather (OTPW) Dataset, original source: https://www.transtats.bts.gov/homepage.asp  
# MAGIC Analysis of the Influence of Factors on Flight Delays in the United States Using the Construction of a Mathematical Model and Regression Analysis: https://ieeexplore.ieee.org/document/9795721
# MAGIC 2. Time Series and Cross Validaition: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20
# MAGIC 3. Parquet files: https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html
# MAGIC 4. 
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
