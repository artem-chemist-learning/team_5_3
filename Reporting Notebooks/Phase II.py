# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Airline Delays: Phase 2

# COMMAND ----------

# MAGIC %md
# MAGIC ##The Team and Phase Leader Designations
# MAGIC <pre>
# MAGIC     Bailey Kuehl             Artem Lebedev              Erik Sambrailo          Lucy Moffitt Herr   
# MAGIC   bkuehl@berkeley.edu  artem.lebedev@berkeley.edu e.sambrail0@berkeley.edu   lherr@berkeley.edu           
# MAGIC     Week 1 Leader             Week 2 Leader            Week 3 Leader             Week 4 Leader                    
# MAGIC <div>
# MAGIC <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/bailey.png?raw=true" width="200"><img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/art.png?raw=true" width="200"> <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/erik.png?raw=true" width="200"> <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/lucy.png?raw=true" width="200"> 
# MAGIC </div><pre>

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Abstract
# MAGIC ### Phase 2 Abstract
# MAGIC Our project aims to leverage machine learning techniques to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. The goal of this work is to advise Delta airlines on the key factors that influence delays so that they can better anticipate potential delays and mitigate them. Similarly to Phase 1, we utilized approximately 75% of the data in the 1-year dataset from the *On Time Performance and Weather (OTPW)* dataset -- an airline and weather dataset containing data from 2015 - 2019 [1]. During Phase 2 of our project, we sought to narrow down our dataset to key predictive features through extensive exploratory data analysis and feature engineering. Additionally, we aimed to train a baseline model on our data by predicting the average delay, which we believe to be a fitting baseline because it is simple in terms of computation and resources, but has room for improvement through more advanced models. Also during this phase, we sought to build and train more complicated models, including logsitic regression and random forest. As described during Phase 1, we have chosen to measure our model performances in terms of precision while maintaining our desired 80% recall, selected for the minimum recall required to be accepted in industry. As such, our baseline model and best logistic regression model resulted in precision values of 24.8% and 31%, respectively. As for our best random forest classifier, chosen for its ability to specify feature importance, we achieved a precision of 28%. Thus, our best modeling pipeline was from our experimentation with logistic regression, which involved adding engineered features including average delay at the origin airport and engineered weather features. For the next and final phase of this project, we hope to iterate on our current models to further improve performance values. Such iterations might include additional feature engineering (such as adding an isHolidayWindow feature), potentially joining an additional dataset to our current data, and fine-tuning existing model parameters through grid search. We hope to optimize our model in order to gain insights about key factors affecting airline delays so that we can share our results with our employer, Delta Airlines, and help them mitigate potential causes for delays before they can occur.
# MAGIC
# MAGIC ### Phase 1 Abstract
# MAGIC Air travel in the United States is the preferred method of transportation for commercial and recreational use because of its speed, comfort, and safety [1]. Given its initial popularity, air travel technology has improved significantly since the first flight took off in 1908 [2]. For example, modern forecasting technology allows pilots to predict the optimal route and potential flight delays and cancellations given forecasted headwinds, storms, or other semi-predictable events. However, previous studies have found that weather is actually not the primary indicator of whether a flight will be delayed or canceled [1]. Today, seemingly unexpected flight delays are not only a nuisance for passengers, but also could a potentially detrimental threat to the airline industry if customers continue to lose trust in public airline capabilities. Thus, the primary goal of this project is to predict flights delays more than 15 minutes in duration that occur within 2 hours prior to the expected departure time. To accomplish this, we will extract airline and weather data spanning the years 2015 - 2019 from the *On Time Performance and Weather (OTPW)* dataset [3]. Feature selection will be performed through null thresholding (dropping features with more than 90% nulls) and lasso regularization. Key features are hypothesized to be Airline (e.g. *Delta, Southwest*), expected maintenence, history of delays for a given flight number (route), and extreme weather (e.g. ice or snow) [4]. We will perform data cleaning, imputation, and exploratory analysis on the remaining data. The cleaned data will be split into test, train, and validation sets via cross-validation on a rolling basis given the time series nature of the data. We will then build and train a logisitic regression model as a baseline, as well as a random forest to predict delays. The proposed pipeline is expected to perform efficiently in terms of runtime given our proposed utilization of partitionable parquet files in place of the more commonly used CSV files. Finally, to measure the success of our model, we propose to use precision and recall, optimizing the tradeoff between the two such that precision is maximized given a goal recall of 80%. Using the results from this project, we hope to advise airlines on key factors affecting flight delays so that they can mitigate them to better satisfy their customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Project Description
# MAGIC ### Data Description
# MAGIC
# MAGIC We began our project utilizing the three datasets depicted below.  
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Preliminary Datasets</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/prelim_data.png?raw=true" width="700">
# MAGIC </div>
# MAGIC
# MAGIC #### Flights  
# MAGIC
# MAGIC The flights dataset is sourced from the TranStats data collection (U.S. Department of Transportation). The full dataset consists of on-time performance data for 31,746,841 U.S. passenger flights between 2015 and 2021, with 109 total features.
# MAGIC Key features to use in our EDA and modeling include flight and carrier identifiers, airport location information, and delay time and source attributes.
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Flight Dataset Features</strong>
# MAGIC     <br>
# MAGIC         <img src="https://github.com/esambrailo/w261/blob/13f7a831ab4e77b37423acbfe83375262fe5609b/flight_data.png?raw=true" width="550">
# MAGIC </div>
# MAGIC
# MAGIC #### Weather
# MAGIC The weather dataset comes from NOAA (National Oceanic and Atmospheric Administration).  After review of their website, various datasets, and documentation, we found that our dataset best aligns with their Local Climatological Data (LCD) dataset.  This dataset consists of hourly, daily and monthly weather observation summaries.  Below is a table showing the general sections of features, some examples from each, and our understanding of those features.  
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Weather Dataset Features</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/weather_features.png?raw=true" width="800">
# MAGIC </div>
# MAGIC
# MAGIC An important discovery from our review of the documentation of this dataset is how the hourly, daily, and monthly values are populated in the dataset. The LCD documenation states: "After each day’s hourly observations, the daily data for that day is given in the following row and monthly data is in the row following the final hourly observations for the last day in the month."*[13]*  To better understand this we reviewed all weather observations for a specific station Los Angeles Airport (LAX) for a specific day 1/10/15.  There were a total of 40 observations recorded that specific day, accross (4) different report types. Each report type had varied observation frequencies and features included. Below is a table showing those details. There was a clear delineation between the records that held hourly vs. daily observations.
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>LAX - 1/10/15: Weather Observations</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/lax_weather_sample_20150110.png?raw=true" width="400">
# MAGIC </div>
# MAGIC
# MAGIC #### OTPW Pre-Joined Dataset
# MAGIC In addition to the separate flight and weather datasets, we have access to a third dataset which is a join of the flight and weather data.  The source and underlying logic used for this dataset was somewhat unknown.  Our intent was to utilize this dataset as much as possible to minimize redundant compute efforts. Our first step was to review the logic used in the join. 
# MAGIC
# MAGIC ##### Join Logic
# MAGIC With our understanding of the separate datasets, and additional EDA of the OTPW dataset, we were able to develop our assumptions for the logic of the join. We concluded that the join was performed in two steps. Below is a diagram depicting the assumed logic.  
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Assumed Join Logic</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/assumed_logic.png?raw=true" width="700">
# MAGIC </div>
# MAGIC
# MAGIC A weather station was joined to each flight record for both the departure and arrival airport locations.  Based on the data, we assume this was performed utilizing the latitude and longitudes for both airports and weather stations. The closest weather station was joined to each respective airport.  The second join combined weather observation records to each weather station.  Weather records appeared to be joined to the flight data using the weather station ID for the departing flight.  What we found somewhat concerning, was how the records appeared to be joined from a time standpoint. A small sample of the time-based components of the joined dataset is presented below. Based off this sample, it appears that the "4 hours prior departure (UTC)" timestamp from flights was used to find the next available corresponding timestamp in the weather data table. 
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Time-Based Components</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/timejoin_sample.png?raw=true" width="700">
# MAGIC </div>
# MAGIC
# MAGIC This join results in flights joined to inconsistent types of weather observations, based on the timing of that specific flight.  Most flights will be joined to hourly records, while others may coincidentally be joined to other reported summaries (like daily summaries).  Additionally, the documentation for the weather data states that timestamps are local time, whereas the weather data is in UTC.  There are a few shortcomings that would come from using this joined dataset as is:
# MAGIC  - **Minimal Daily Features** - There are not enough records with daily summary information to utilize any of those features. 
# MAGIC  - **No Arrival Airport Weather** - Although departure weather is more pertinent, there is no arrival airport weather included. 
# MAGIC  - **Stale Weather Observations** - The mis-alignment in time standards used results in joined hourly weather records that pre-date flights by upwards of 11 hours (continental US)
# MAGIC  - **Limited Weather Feature Engineering Opportunities** - Any further time series weather feature engineering will be biased by what flight records exist. 
# MAGIC
# MAGIC With these concerns, we chose to start over on the join of weather features.  In the next section we speak to the initial workflow diagram, which includes the re-processing of the raw weather data. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA & Feature Selection Pipeline
# MAGIC The below diagram depicts our overal work flow for the first half of our data pipeline.  As can be seen from the diagram, we are performing multiple iterations of EDA, feature selection, and feature engineering.  We will speak in depth to each of these iterations in the following sections. 
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Data Pipeline Workflow</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/data_workflow.png?raw=true" width="400">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Raw Weather Data Scope
# MAGIC
# MAGIC Re-joining the raw weather data gave us an opportunity to perform EDA and feature engineering prior to joining with the flight data.  Below is a more detailed diagram of the steps and logic used for this scope. Asterisks in the diagram indicate the locations of data checkpoints.  
# MAGIC <br>*(The work for these steps can be found in the series of notebooks located here: https://adb-4248444930383559.19.azuredatabricks.net/browse/folders/1128664627674437?o=4248444930383559. For this section, the notebook number of the associated work will be referenced as (wb#).)*
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Raw Weather Workflow Detail</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/58f89dffe186b37df3a7effbb8e201d3c87840fa/raw_weather_workflow.png?raw=true" width="400">
# MAGIC </div>
# MAGIC
# MAGIC ##### Dataset Reduction & Split
# MAGIC With the weather observation dataset being of substantial size, the first step was to reduce it to only include the weather stations that correspond to airports. This was derived by quantifying all the unique weather stations represented in the entire OTPW dataset *(wb1)*.  Reducing the weather data to only these stations dropped the number of unique weather stations to 370 *(wb2)*.  Next, we split the raw weather into to multiple datasets, corresponding to hourly observations, and daily and monthly summaries.  We only focused on the Daily and Weather features, and chose to ignore monthly.  
# MAGIC
# MAGIC ##### Researching Weather's Effect on Flights
# MAGIC Before exploring our data further, we researched the effects of various weather conditions on commercial airlines.  The general findings are that commercial flights are fairly resilient to a variety of weather conditions, but extreme conditions can still be challenging.*[6]*  Below we highlight some of the weather conditions that pose challenges for commercial airlines.
# MAGIC
# MAGIC      
# MAGIC - **Strong Cross-Winds**: While some wind is actually preferred for takeoff and landing, it can become problematic if it is a cross-wind.  Cross-winds that exceed 40 mph can be challenging for takeoff and could potentially delay a flight.*[6]*
# MAGIC - **Poor Visibility**: Planes are relatively unaffected by precipitation and clouds.  It is the poor visibility that can accompany this weather that is the biggest challenge for flights. Especially near and on the ground.*[7]* 
# MAGIC - **Thunderstorms**: While planes can technically take off in thunderstorms, generally airlines wait until the storm subsides before departing.*[10]*
# MAGIC - **Snow/Icing**: Icy conditions can be problematic for runways, and can require additional plane de-icing procedures before flights.*[7]*
# MAGIC - **Reduced Air Density**:  Air density has a direct affect on a plane's ability to lift.  Different weather conditions, such as hot weather can reduce air density and a planes ability to lift. Planes get 1% less lift with every 5.4 degrees Fahrenheit (3 degrees Celsius) of temperature rise. *[9]*  While the difference is measurable, its not enough, by itself, to prevent a commercial airline from taking off.
# MAGIC
# MAGIC
# MAGIC ##### Weather EDA
# MAGIC
# MAGIC We performed EDA on both the Daily and Hourly Datasets.  With our research findings indicating that commercial flights are resilient to weather, we chose to be aggressive in our preliminary feature reduction.  For both the Daily and Hourly datasets, we kept only metric features *(wb3&4)*. More complex features, such as "sky condition", would have required additional parsing logic.  We made the educated assumption that sky condition would not provide enough additional insight, beyond the metric features, to justify the additional time and compute resources necessary to include. 
# MAGIC
# MAGIC We compiled groups of similar features.  For each grouping, we calculated the Pearson Correlation Coefficient accross the entire train dataset (2015-2018), and produced a pair plot for a small sampling of each. Below is an example of the calculated coefficients and sample pairplots for the metric temperature features found in the Hourly weather data. *(The results of all grouped features can be found in the notebooks (nb3&4))*
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Hourly Temperature (F) Features Pairplot</strong>
# MAGIC     <br>
# MAGIC     3,000 Random Sample from (2015-2019)
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/pairplot.png?raw=true" width="700">
# MAGIC </div>
# MAGIC
# MAGIC **Pearson Correlation Calculations for Hourly Temperature Features** *(Accross entire train dataset.)* 
# MAGIC |Feature Pair|Pearson Correlation Coeff.|
# MAGIC |---|---|
# MAGIC |'HourlyDewPointTemperature', 'HourlyDryBulbTemperature'|0.8313|
# MAGIC |'HourlyDewPointTemperature', 'HourlyWetBulbTemperature'|0.8671|
# MAGIC |'HourlyDryBulbTemperature', 'HourlyWetBulbTemperature'|0.8733|
# MAGIC
# MAGIC As expected, all features are highly correlated.  For each grouping like this, one or two features were selected based on: domain knowledge, number of nulls, and the correlation with the other variables *(wb3&4)*.  In this particular instance the Dry Bulb temperature was selected because it had the least amount of nulls, and because humidity was also an available feature. Wet-bulb and dew-point are derived from dry-bulb and humidity.
# MAGIC
# MAGIC Once the feature selections were made, the Pearson Correlation coefficients and pairplots were calculated again to review the correlations accross groups *(wb3&4)*.
# MAGIC
# MAGIC ##### Weather Feature Engineering
# MAGIC
# MAGIC We performed feature engineering on both the Daily and Hourly Datasets.  For the Daily dataset, we simply added a new feature that represented the next day, which was used to join to the flight data.  The feature engineering for the hourly data was much more involved. 
# MAGIC
# MAGIC ###### Hourly Timestamp Features
# MAGIC
# MAGIC As previously mentioned, the documentation for the weather data stated that all timestamps were reported in local time zones.  The larger OTPW datasets only had timestamps in UTC. In order to properly align weather data for both the departure and arrival locations at a specific time, all weather data needed to be converted to UTC.  We spent some time researching how to derive the UTC for a location based from longitutude and latitude, but were unsuccessful at calculating it directly.  Instead, we extracted the timezone to UTC offsets that existed in the smaller OTPW datasets, where both timestamps were present. *(wb1)*  For any weather station that was not present in that dataset, we imputed the UTC offset that existed for the state of that station. *(wb1)*  Because of daylight savings time, and some states residing in multiple time zones, this process did not result in a dinstinct list.  For the station-specific UTC offsets, we selected the greater offset value, as that represented standard local time.  For the states used for imputting we also selected the larger offset, which represents the standard local time for the most western timezone for a given state.  These UTC offset values were mapped to the Hourly Dataset, and used to convert the local time to UTC time. A more precise conversion of timestamps would have been preferred, but this was the best approach we were able to complete given our time and resource constraints. 
# MAGIC
# MAGIC ###### Hourly Weather Observation Averages
# MAGIC
# MAGIC The hourly weather features only represent what had occured within an hour.  In some cases, like temperature, it is a single snapshot measurement taken. For other metrics, like precipitation, it is the accumulation over one hour.  Regardless of the specifics of the measurement, it only represents a very small window.  A single hourly metric does not provide enough insight to what weather conditions have occurred.   As a hypothetical example,  a hourly precipitation record may be blank, stating that there has been no precipitation in the past hour.  This record could follow a full day of sunshine, or a day that saw 6 inches of rain that resided the hour prior to the recorded measurement.  For this reason, we chose to derive multiple observation windows.  To bridge the gap between the daily summary features from the prior day, and the most recent hourly features, we used 3, 6 and 12 hour windows.  We calculated the average value for each feature by station accross the specified window.  We selected averages as our metric because of inconsistent frequencies of recordings accross weather stations.  Some stations only recorded hourly, while others did every 15 minutes.  A metric like 'accumulated precipitation' is more interpretable than 'average hourly precipitation accumulation', but either works for modeling purposes.  
# MAGIC
# MAGIC In hindsight, min. and max. values should have been derived as well.  Without these, there is a minor gap in our features.  Going back to the hypothetical precipitation feature, if a station saw significant rain in a three hour window, but then nothing for the nine hours that followed, the only metric that would capture this event would be the 12 hour window.  In that twelve hour window, the intensity of the measurement would be "watered down" by the mild measurements of the following 9 hours.
# MAGIC
# MAGIC ##### Joining Weather to OTPW Dataset
# MAGIC
# MAGIC Below is a diagram depicting the logic that was used for joining both the Daily and Hourly datasets to the OTPW dataset. *(wb5)*  Due to the estimated nature of our UTC timestamps in the weather dataset, we created a new feature, 'three_hours_prior_to_departure' from which to join the weather observations. *(wb5)* This additional hour buffer ensures that we do not violate the requirement of making predictions two hours prior to departure. 
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Weather Data Join Detail</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/join.png?raw=true" width="700">
# MAGIC </div>
# MAGIC
# MAGIC The daily weather data can be interpretted as a summary of the weather the day before the flight.  The hourly weather features can be interpreted as various weather observation windows up until approximately three hours before a flight. We now move forward with the next phase of EDA and featuring engineering using this newly joined dataset. 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Newly Joined Data Scope
# MAGIC
# MAGIC The figure below provides a overview of the workflow we followed in performing exploratory data analysis and data cleaning on the newly-joined 1-year OTPW dataset. (The detailed work for these steps can be found in the series of notebooks "01_Phase2_OTPW_EDA" through "04_Phase2_OTPW_EDA" located here: https://adb-4248444930383559.19.azuredatabricks.net/browse/folders/1128664627674483?o=4248444930383559.
# MAGIC <br>
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Joined Data Workflow Detail</strong>
# MAGIC     <br>
# MAGIC     <br>
# MAGIC         <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/eda_process_chart.png?raw=true" width="600">
# MAGIC </div>
# MAGIC
# MAGIC In this process, our priorities were as follows: 
# MAGIC 1. Addressing missing values: in reviewing features with high proportions of missing values, our goal was to impute missing values based on background research wherever possible, and to remove observations only when necessitated by low conceptual relevance to flight delay prediction and/or time constraints on implementing more sophisticated imputation methods.
# MAGIC 2.  Feature selection: when comparing similar features to select the more useful predictor and outcome variables, the general rule of thumb was to use the most granular measure available. Whenever possible, any features that appeared to have limited relevance or redundant information were evaluated against other similar features for comparison. For example, in comparing the `TAXI_IN` and `TAXI_OUT` feature values to the calculated differences between `ARR_TIME` and `WHEELS_ON` or `DEP_TIME` and `WHEELS_OFF` (respectively), we determined that we could safely drop the `WHEELS_OFF` and `WHEELS_ON` columns without losing information.  By concentrating first on removing any features determined not to contribute additional information to prediction, our aim was to increase the efficiency of working with the data in EDA and modeling with an eye toward iterating the process on the 5-year data. Whenever purely descriptive features (e.g., airport names) weren't useful for modeling but had vlaue for visualization and reference, we created and stored look-up tables for future reference.
# MAGIC 3.  Imputation: when possible, we imputed remaining missing values based on domain input. We also removed subsets of observastions (e.g., cancelled flights, which did not have associated delay data) based on the rationale that these could not contribute predictive value to our models at this stage. 
# MAGIC
# MAGIC The following table summarizes each the individual steps in the process of exploring and cleaning the 1-year data, as well as their impacts on the dimensionality of the dataset where applicable. 
# MAGIC <br>
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC         <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/flights_eda_summary_table.png?raw=true" width="900">
# MAGIC </div>
# MAGIC
# MAGIC Despite our best efforts, given the need to prepare data for our models as soon as possible, there are a few decisions we hope to revisit in the final phase of this project. In particular, determining ways to conserve more of the weather observations with missing values via imputation would be more optimal in the next round. 
# MAGIC <br>
# MAGIC <br>
# MAGIC #### EDA of the Newly Joined Data: Key visualizations
# MAGIC
# MAGIC **Flight Delay Time Distribution**<br>
# MAGIC The frequency distribution of all domestic U.S. flights in 2015 (for all airports) is highly skewed, with a substantial proportion of negative delays Simple visualizations such as this point to the need to understand which characteristics are unique to the small proportion of flights with extremely long delay times. 
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <br>
# MAGIC         <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/total_flights_by_depdelay_group2.png?raw=true" width="600">
# MAGIC </div>
# MAGIC
# MAGIC **Seasonal Trends in Flight Delays**<br>
# MAGIC We also anticipated some seasonal trends in average flight delays, which are represented in more detail at the day-of-year level than on a monthly or quarterly level. We expected to observe spikes in delays when air travel increases in mid-summer and winter holiday season, but the variation within each month was somewhat surprising.  
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <br>
# MAGIC         <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/avg_departure_delay_by_day_of_yeat.png?raw=true" width="800">
# MAGIC </div>
# MAGIC
# MAGIC **Carrier Variations in Flight Delays**<br>
# MAGIC While we were not surprised to encounter differences in average delays by airline (carrier), the level of variation across carriers helps to underscore the importance of carrier-based features in our models. 
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <br>
# MAGIC         <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/average_dep_delay_by_carrier.png?raw=true" width="800">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC **Weather Feature Families: Relationships to Delay Times**<br>
# MAGIC With regard to the weather features, many featured moderate to high Pearson's correlation values with delay times as well as weather features within the same family of measurement, particularly those closest in timeframe. The correlations heatmaps below for precipitation and ___ features, respectively, demonstrate the utility of selecting from among the highly-correlated weather features, but they also show the increasing relationship to delays as timepoints move closer to departure. Note as well the extremely high overall Pearson's correlation between arrival delay time and departure delay time (0.94). 
# MAGIC <br>
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Precipitation Features</strong>
# MAGIC     <br>
# MAGIC     <br>
# MAGIC         <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/precipitation_heatmap.png?raw=true" width="600">
# MAGIC </div>
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Visibility Features</strong>
# MAGIC     <br>
# MAGIC     <br>
# MAGIC         <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/visibility_heatmap.png?raw=true" width="600">
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling Pipelines
# MAGIC ### Summary of Steps
# MAGIC During Phase II of this project, our modeling pipeline began after all exploratory data analysis was performed. The first step of modeling involved splitting the data into folds in order to avoid leakage. We designated the data from the year 2019 as a "held-out" test set which we will not used until all model training and experimentation is completed and a final best model is selected. To further avoid data leakage, we performed cross-validation given that our data is time series. 
# MAGIC
# MAGIC Next, we began building our models. Our first model was a statistical baseline sans machine learning techniques. The One note regarding this process is that we initially did not account for class imbalance prior to training and evaluation, so this step was added recently and has yet to be fully refined. For this model, we computed the average departure delay at the departure location and use this as proxy for delay of a given flight. Additionally, we built logistic regression and random forest models (please see *Models and Performance* section for model configuration details).
# MAGIC
# MAGIC After we built our models, we experimented by training different variations or each model, such as including different subsets of features (based on feature importance evaluations), modifying modeling parameters (e.g. numTrees for random forest), and testing regularization techniques. This step was fairly iterative so that we could see the effects from each experiment and work towards increasing our performance metrics. Note that each experiment was evaluated on both the training and validation sets, but the held-out set mentioned earlier has yet to be evaluated on.
# MAGIC
# MAGIC Finally, we selected the models that resulted in the best performances during experimentation as our "best model" for logistic regression and random forest.
# MAGIC
# MAGIC A diagram of our full modeling pipeline described above is shown below.
# MAGIC <br/>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Modeling Pipeline Workflow</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/baileykuehl/261_usage/blob/main/modeling_pipeline.png?raw=true" width="500">
# MAGIC </div>
# MAGIC
# MAGIC ### Cross Validation and Class Imbalance
# MAGIC All models were trained using overlapping blocks for cross validation illustrated below. The test/train split was 4/1 and we decided to overlap test and train sections of the blocks to make sure the model is trained on the full yearl of data. The leak that arises from using the same data for testing and training is mitigated by the fact that it happens in different models.
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Cross-Validation Split</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/Cross-validation.png?raw=true" width="400">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC ### Experiment Details
# MAGIC Below is a summary of the experiments performed on our set of models. Note that the "Build Time" includes the number of minutes taken to assemble feature vectors, build any necessary pipelines, and finish training. To contextualize these times, the cluster information from each experiment is also provided. Please also note: "All OTPW features" excludes features removed during our EDA.
# MAGIC |#|Experiment|Description|Family of Input Features|# Input Features|Cluster|Build Time (min)|
# MAGIC |---|---|---|---|---|---|---|
# MAGIC |1|Baseline|Statistical average (no ML)|Flight delay for airport <br/><br/>Average over -4 to -2 hours|2|16 GB Memory <br/><br/>4 Cores|0|
# MAGIC |2|Trivial Logistic Regression|Simplest ML model, all features|Select OTPW features|13|16 GB Memory <br/><br/>4 Cores|18|
# MAGIC |3a|Engineered Logistic Regression (1 year)|Some engineered features|Average delays over various windows|7|16 GB Memory <br/><br/>4 Cores|28|
# MAGIC |3b|Engineered Logistic Regression (4 year)|Some engineered features|Average delays over various windows|5|16 GB Memory <br/><br/>4 Cores|35|
# MAGIC |4|Random Forest|Simplest AI model, no engineered features|Select OTPW features|6|16 GB Memory <br/><br/>4 Cores|4-15**|    
# MAGIC * **Build time was dependent on model parameters used.
# MAGIC
# MAGIC ### Success Metrics and Loss Functions
# MAGIC To measure the performance of the model, we will use precision at 80% recall (see Phase 1 for detailed justification). The equations for each of these metrics are listed below: 
# MAGIC
# MAGIC $$Precision = \frac{\text{correctly predicted as delayed}}{\text{all flights predicted as delayed}} = \frac{\text{True Positives}}{\text{True Positives + False Positive}}$$
# MAGIC
# MAGIC $$Recall = \frac{\text{correctly predicted as delayed}}{\text{all flights actually delayed}}= \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$$
# MAGIC
# MAGIC <br>
# MAGIC The models we used during this phase include Logistic Regression and Random Forest. For logistic regression, we utilize cross entropy, also known as the log (logarithmic) loss, as our loss function. This loss function is chosen because it quantifies the difference between predicted probabilities and actual values, making it an important classification metric [4]. 
# MAGIC
# MAGIC For our other model, we know that Random Forest doesn't explicitly optimize a loss function like other gradient-based methods(e.g. logistic regression). However, Spark's RF classifier uses *Gini impurity* as the default impurity measure for splitting nodes during tree construction. In this context, Gini impurity is used to evaluate the impurity of a node. It quantifies the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled based on the class labels in the node [8]. These equations relevant to our models are defined below:
# MAGIC
# MAGIC
# MAGIC $$Cross Entropy Loss = \-\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]\$$
# MAGIC
# MAGIC $${Gini} = 1 - \sum_{i=1}^{C} p_i^2$$

# COMMAND ----------

# MAGIC %md
# MAGIC # Models and Performance

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment 1: Baseline Models (1 year data)
# MAGIC Link to Baseline code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1198761045465243
# MAGIC
# MAGIC These two models place performance of more sophisticated algorithms in context.
# MAGIC
# MAGIC #### Random Guess
# MAGIC Random guess, the absolutely simplest model, yields approximately 18% precision at any level of recall. This is becasue the dataset is heavily biased to the on-time flights, and only ~18% of the flights are delayed. The probaility of the guess controld the recall: if we label the flight delayed with 50% probability, we mislabel 50% of the delayed flights.
# MAGIC
# MAGIC #### Arithmetic Mean
# MAGIC A slightly more complex model only requires a glance at the departure table of the airport 2 hours prior to deprture. For this model we calculate the average departure delay at the origin, and assume that the flight in question will have the same delay. We can label the flight as "Predicted Delayed" if the predicted delay crosses a certain cut-off. At cut_off of ~2 min we achived 80% recall and ~25% precision. in other owrds, if the average delay at the airport, based on the data avaliable 2 hours ahead of departure, exceeds 2 min, then we call the flight delayed. This gave only is a slight improvement over random guess at the lower recalls, but demonstrated a huge improvement at the low recall levels (likely more relevant for passengers that for airlines)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment 2: Trivial Logistic Regression (1 year data)
# MAGIC
# MAGIC Link to Baseline code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1871383191021758
# MAGIC
# MAGIC This model was meant to demonstrate the potential of the simplest Machine Learning algorythm with no additional input from a Data Scientist. We also used this model to do preliminary featyre selection for further exploration in subsequenst models. All avaliable features were used in the model, with categorical features encoded as dense verctors. In order to select features, we used Lasso regualrization with regularization parameter 0.001. At this point, the following features were selected:
# MAGIC  Feature|Weight
# MAGIC  --|--
# MAGIC  Is it Thursday?|0.02657
# MAGIC  Airline UA?|0.00718
# MAGIC  Airline B6?| 0.01426
# MAGIC  Airline NK?| 0.028558
# MAGIC  Is it August?| 0.041008
# MAGIC  Origin Daily Precipitation| 0.0017647
# MAGIC  Origin Daily Average RelativeHumidity| 0.028534
# MAGIC  Origin Daily Sustained Wind Speed| 0.01917
# MAGIC  Origin Daily Average Relative Humidity| 0.00471
# MAGIC  Origin Hourly Dry Bulb Temperature| 0.21057
# MAGIC  Origin Hourly Precipitation| 0.03769
# MAGIC  Dest Hourly Dry Bulb Temperature| 0.15246
# MAGIC  Dest Hourly Precipitation| 0.010683
# MAGIC
# MAGIC  For categorical variables the following categories are baseline: Day of the week: Saturday, Carrier: HA, Month: February. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment 3a: Engineered Logistic Regression (1 year data)
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
# MAGIC  Origin Yesterday Snowfall| 0.0130466
# MAGIC  Origin Precipitation over 6H window| 0.013532
# MAGIC  Origin Precipitation over 12H window| 0.003369
# MAGIC  Average delay at the origin| 0.19865
# MAGIC  Previous delay of the aircraft| 0.470280
# MAGIC  Average delay of the carrier| 0.22544
# MAGIC  Number of flights from the origin| 0.148409

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment 3b: Engineered Logistic Regression (4 year data)
# MAGIC
# MAGIC Link to Baseline code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1537154891793919
# MAGIC
# MAGIC This model was meant to test the computational performance of our approach over the large dataset. The following features ended up being significant
# MAGIC  Feature|Weight
# MAGIC  --|--
# MAGIC  Origin Precipitation over 12H window| 0.001861
# MAGIC  Average delay at the origin| 0.30613
# MAGIC  Previous delay of the aircraft| 0.494963
# MAGIC  Average delay of the carrier| 0.34518
# MAGIC  Number of flights from the origin| 0.17765

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment 4: Simple Random Forest (1 year data)
# MAGIC
# MAGIC Link to RF code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/2798664673132331/command/2798664673132333
# MAGIC
# MAGIC Our final model was a simple Random Forest Classifier. After performing a feature importance evaluation, te following features had non-zero importances:
# MAGIC
# MAGIC  Feature|Weight
# MAGIC  --|--
# MAGIC DISTANCE| 0.6567227824
# MAGIC origin_HourlyDryBulbTemperature	| 0.143551
# MAGIC origin_HourlyStationPressure |	0.105756
# MAGIC origin_HourlyPressureChange	| 0.074535
# MAGIC origin_HourlyWindGustSpeed	| 0.013029
# MAGIC origin_HourlyWindDirection	| 0.00640
# MAGIC
# MAGIC <br>
# MAGIC The above features were passed into our model for training. Within this model, we performed a series of experiments, including varying the parameter values to the classifier. The experiments are shown in the table below.
# MAGIC
# MAGIC
# MAGIC  Model|numTrees|maxBins|maxDepth|minInstancesPerNode|Precision|Recall
# MAGIC  --|--|--|--|--|--|--|
# MAGIC 1| 10|5|3|3|0.68926|0.83021
# MAGIC 2| 100|5|3|3|0.68926|0.83021
# MAGIC 3| 100|20|3|3|0.68926|0.83021
# MAGIC 4| 25|10|5|1|0.68926|0.83021
# MAGIC **5| 50|20|10|2|0.79986|0.83021
# MAGIC
# MAGIC Note that the metrics reported are for weighted precision and weighted recall, which are showing inflated values compared to our reported performance. This is because of our class imbalance, which we plan to account for more optimally in the final phase. It is also worth noting that changing only the number of trees (row 1 to row 2) did not have an effect on the resulting performance; however, this did significantly increase runtime. Experiment 5 performed the best in terms of precision, likely due to the increase in tree depth compared to the other experiments. A more refined approach will be to use a grid search to obtain the best model and its resulting parameters.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results and Discussion
# MAGIC ### Experiment Summary: Comparative performance of models
# MAGIC
# MAGIC Combining all of the experiments from above, we summarize the experiments and their performance in the table below. Note that the  reported performance metric is precision at 80% recall on test data (and training data, where applicable). 
# MAGIC
# MAGIC  Model|Training time (min) | Dataset | Test performance (%) | Train Perfomance (%)
# MAGIC  --|--|--|--|--
# MAGIC  Random Guess| 0| 1 Year |19.7|-
# MAGIC  Baseline| 0 | 1 Year |24.8|-
# MAGIC  Trivial LR| 18 |4 Years |20.2|22.7
# MAGIC  Engineered LR| 39| 4 Years |31|32
# MAGIC  Final LR| 18 | 4 Years |24.6|-
# MAGIC  Random Forest| 3-10 | 4 Years |28|57.1
# MAGIC
# MAGIC From the above data, we find that our best model is the Engineer Logistic Regression. It is initially surprising that this performs better than Random Forest; however, we believe this to be related to the features used to train each model. For example, we constructed a simple random forest model with limited features, all of which came directly from the OTPW dataset and none of which came from our team's feature engineering. Additionally, due to time constraints, grid search to select the optimize random forest model was not completed during this phase.
# MAGIC
# MAGIC One alamring finding from this table is that the training performance on random forest is significantly greater than the test performance. This might indicate our model is overfit, but may also indicate other issues related to class imbalance, explaiend in further details below. A visual example of the difference between training and test performance is shown below for Trivial logistic regression. We note that the performance is very similar, indicating absence of overfitting. However, these values in general are low, which indicated that there was room for improvement in this model.
# MAGIC
# MAGIC  <br>
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Training vs Test Performance: Logistic Regression </strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/Test vs train.jpg?raw=true" width="500">
# MAGIC </div>
# MAGIC
# MAGIC The image below illustrates comparative performance of all LR, RF, and Baseline models for various decision thresholds. It is worth noting that at some threshold all models perform no better than random guess, but if the user is willing to sacrifice the recall, up to 70% precision can be achieved. It truly depends on the use scenario what decision threshold to chose. At the level of usable 80% threshold, the most sophisticated engineered LR model only achieves 31% Precision, which is hardly informative for the airline.
# MAGIC
# MAGIC  <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Model Performance Comparison: Precision at 80% Recall </strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/baileykuehl/261_usage/blob/main/model_perform.png?raw=true" width="500"> 
# MAGIC </div>
# MAGIC
# MAGIC From the figure, we can observe that our aggressively feature-selected 4-year logistic regression model did not actually outperform the same model on the 1-year dataset. It is also interesting that trivial logistic regression that does not incorporate any engineered features, yet performs worse than the baseline model.
# MAGIC This observation gives us hope that the jobs for Data Scientists will not be eliminated by automated code-writing systems in the foreseeable future. 
# MAGIC
# MAGIC Through experimentation, we found dataset imbalance is a concern, given that most flights are not delayed. We attempted to address this concern by dropping ~82% of on-time flights from the train deta set. This produced a balanced training data, while keeping test data relevant. Engineered model was trained on these two datasets (with no cross-validation) to investigate effectiveness of this technique. The graphs below present performance of the models at various decision thresholds. It is clear that performance remains virtually unchanged, given that the approptiate decision threshold is chosen. For the model trained on the imbalanced set the threshold is close to 20%, while for the model trained onthe balanced set, it is near 45%. We conclude that dropping random on-time records might be beneficial from the compute perspective, but does not yield a better model.  
# MAGIC
# MAGIC  <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Model Performance with Corrected Class Imbalance </strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/Balancing.jpg?raw=true" width="500"> 
# MAGIC </div>
# MAGIC  <br>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and Final Thoughts
# MAGIC The purpose of this project is to use machine learning to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. This work has important implications for the general public because flight delays are extremeley disruptive for both commercial and leisure passengers, flight crew, and other airline employees. We hypothesized that using machine learning modeling techniques, including logistic regression and random forest, would provide a precise, reliable methodology for predicting flight delays. To account for factors outside of those provided in our dataset, we contributed additional features through careful engineering, including average delay at the origin airport, time windows of weather features for the previous flight, average airline carrier delay, and more. Additionally, we contribute a custom-joined dataset containing weather and airline features, which expands upon the available data from the original OTPW dataset. Using these features and dataset, we were able to achieve a maximum precision of 31% at a recall of 80% on our engineered Logistic Regression model.
# MAGIC
# MAGIC While these results are not optimal in terms of performance, they are still significant in that we have proven that flight prediction can in fact be modeled through machine learning. Nevertheless, we feel confident that we can further improve our models by:
# MAGIC 1. Fine-tuning model parameters (e.g. "numTrees", "maxDepth" for Random Forest)
# MAGIC 2. Additional feature engineering & experimentation (potentially adding an outside dataset)
# MAGIC 3. Non-linear precipitation, interaction terms, graph-based, time-based delay propagation
# MAGIC
# MAGIC For the final stage of our project, one challenge we anticipate with our proposed improvements is attempting to engineer features through obtaining additional data: it may be difficult to find data compatible with our current dataset or to generate desired features (such as a proxiy for the current economic status of each airline carrier). We also anticipate difficulty in optimizing the time required to utilize the full dataset on our model.

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
# MAGIC 4. Log loss: https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/
# MAGIC 5. Chat GPT for code optimization: https://chat.openai.com/
# MAGIC 6. Pilot Institute. (2022, January 26). The effect of wind speed on an airplane. https://pilotinstitute.com/wind-speed-airplane/#:~:text=The%20only%20thing%20a%20strong,flight%20takes%20longer%20than%20expected. 
# MAGIC 7. Pilot Institute. (2022b, December 3). Can planes fly in rain - or other severe weather? https://pilotinstitute.com/can-planes-fly-in-rain/ 
# MAGIC 8. Random Forest Parameters: https://towardsdatascience.com/random-forest-regression-5f605132d19d
# MAGIC 9. Prisco, J. (2023, July 22). Why high temperatures can make planes too heavy to take off. CNN. https://www.cnn.com/travel/article/climate-change-airplane-takeoff-scn/index.html#:~:text=%E2%80%9CLift%20depends%20on%20several%20factors,of%20temperature%20rise%2C%20Williams%20said. 
# MAGIC 10. Is it safe to fly a plane in a thunderstorm?. AirAdvisor. (n.d.). https://airadvisor.com/en/blog/is-it-safe-to-fly-a-plane-in-a-thunderstorm#:~:text=Can%20a%20plane%20take%20off,Fly%20With%20an%20Ear%20Infection%3F 
# MAGIC 11. Central, W. S. (n.d.). Secret law of storms. World Storm Central - world storm watch and all about storms. https://www.worldstormcentral.co/law%20of%20storms/secret%20law%20of%20storms.html#:~:text=A%20storm%20also%20typically%20requires,1009%20hPa%20(or%20mb). 
# MAGIC 12. PhysLink.com, A. S. (n.d.). How does humidity effect the way that an airplane flies? https://www.physlink.com/education/askexperts/ae652.cfm 
# MAGIC 13. Local climatological data (LCD). National Centers for Environmental Information (NCEI). (2023, November 8). https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data 
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC # Appendix

# COMMAND ----------

# DBTITLE 0,Appendix
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
# MAGIC       <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/Gant_%26_Credit_Plan.png?raw=true" width="1600">
# MAGIC    </div>
# MAGIC   </body>
# MAGIC </html>
# MAGIC <br>

# COMMAND ----------


