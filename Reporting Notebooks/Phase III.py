# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Airline Delays: Phase 3

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
# MAGIC ### Phase 3 Abstract
# MAGIC Our project aims to leverage machine learning techniques to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. The goal of this work is to give Delta airlines a tool to flag a flight as a potentially delayed so that they can better anticipate potential delays and mitigate them. To narrow down our dataset to key predictive features, we performed extensive exploratory data analysis, rejoined the *On Time Performance and Weather (OTPW)* dataset [1], and used feature selection via setting a null threshold and Lasso Regularization. We built, trained, and evaluated logsitic regression, random forest, and multilayer perceptron (MLP) models. Finally, we experimented with combining these three models to create an ensemble model. To measure model performance, we chose to evaluate precision at a threshold of 80% recall, which we belive would be the minimum recall of an actianble prediction. We threshold our predictions such that the resulting recall is approximately 80% and strive to achive the highest precision possible at this threshold. Our best pipeline was our ensemble model, which demonstrated 26.5% precision on the test dataset. This pipeline was a result of fine-tuning the hyperparameters within each respective model, such as numTrees and maxDepth in Random Forest, and included engineered features (e.g. average delay at the origin airport, hourly precipitatation, etc.). As a result of this project, we contribute our joined dataset, models and corresponding hyperparameters, and recommendations for future work.
# MAGIC
# MAGIC ### Phase 2 Abstract
# MAGIC Our project aims to leverage machine learning techniques to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. The goal of this work is to advise Delta airlines on the key factors that influence delays so that they can better anticipate potential delays and mitigate them. Similarly to Phase 1, we utilized approximately 75% of the data in the 1-year dataset from the *On Time Performance and Weather (OTPW)* dataset -- an airline and weather dataset containing data from 2015 - 2019 [1]. During Phase 2 of our project, we sought to narrow down our dataset to key predictive features through extensive exploratory data analysis and feature engineering. Additionally, we aimed to train a baseline model on our data by predicting the average delay, which we believe to be a fitting baseline because it is simple in terms of computation and resources, but has room for improvement through more advanced models. Also during this phase, we sought to build and train more complicated models, including logsitic regression and random forest. As described during Phase 1, we have chosen to measure our model performances in terms of precision while maintaining our desired 80% recall, selected for the minimum recall required to be accepted in industry. As such, our baseline model and best logistic regression model resulted in precision values of 24.8% and 31%, respectively. As for our best random forest classifier, chosen for its ability to specify feature importance, we achieved a precision of 28%. Thus, our best modeling pipeline was from our experimentation with logistic regression, which involved adding engineered features including average delay at the origin airport and engineered weather features. For the next and final phase of this project, we hope to iterate on our current models to further improve performance values. Such iterations might include additional feature engineering (such as adding an isHolidayWindow feature), potentially joining an additional dataset to our current data, and fine-tuning existing model parameters through grid search. We hope to optimize our model in order to gain insights about key factors affecting airline delays so that we can share our results with our employer, Delta Airlines, and help them mitigate potential causes for delays before they can occur.
# MAGIC
# MAGIC ### Phase 1 Abstract
# MAGIC Air travel in the United States is the preferred method of transportation for commercial and recreational use because of its speed, comfort, and safety [1]. Given its initial popularity, air travel technology has improved significantly since the first flight took off in 1908 [2]. For example, modern forecasting technology allows pilots to predict the optimal route and potential flight delays and cancellations given forecasted headwinds, storms, or other semi-predictable events. However, previous studies have found that weather is actually not the primary indicator of whether a flight will be delayed or canceled [1]. Today, seemingly unexpected flight delays are not only a nuisance for passengers, but also could a potentially detrimental threat to the airline industry if customers continue to lose trust in public airline capabilities. Thus, the primary goal of this project is to predict flights delays more than 15 minutes in duration that occur within 2 hours prior to the expected departure time. To accomplish this, we will extract airline and weather data spanning the years 2015 - 2019 from the *On Time Performance and Weather (OTPW)* dataset [3]. Feature selection will be performed through null thresholding (dropping features with more than 90% nulls) and lasso regularization. Key features are hypothesized to be Airline (e.g. *Delta, Southwest*), expected maintenence, history of delays for a given flight number (route), and extreme weather (e.g. ice or snow) [4]. We will perform data cleaning, imputation, and exploratory analysis on the remaining data. The cleaned data will be split into test, train, and validation sets via cross-validation on a rolling basis given the time series nature of the data. We will then build and train a logisitic regression model as a baseline, as well as a random forest to predict delays. The proposed pipeline is expected to perform efficiently in terms of runtime given our proposed utilization of partitionable parquet files in place of the more commonly used CSV files. Finally, to measure the success of our model, we propose to use precision and recall, optimizing the tradeoff between the two such that precision is maximized given a goal recall of 80%. Using the results from this project, we hope to advise airlines on key factors affecting flight delays so that they can mitigate them to better satisfy their customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Write-Up
# MAGIC ###Proposal
# MAGIC Delta Airlines is a leading airline carrier that periodically grapples with challenges related to flight delays impacting customer satisfaction and operational efficiency. To help Delta tackle this issue, our Data Science team conducted a thorough analysis using machine learning techniques to predict delays exceeding 15 minutes within 2 hours of scheduled departure. Our aim is to furnish Delta Airlines with actionable insights, enabling proactive mitigation of the damage casued by the delay and disruption reduction. Our technical approach involved comprehensive data exploration, feature engineering, and the utilization of diverse machine learning models (logistic regression, random forest, and multilayer perceptron) to achieve a precise prediction of flight delays. Our methodology stands as a reliable tool for predicting and managing these delays effectively.
# MAGIC
# MAGIC ###Impact
# MAGIC #####Customer Satisfaction
# MAGIC Our foremost goal is to leverage historical flight data and weather information to construct predictive models accurately anticipating delays. By discerning crucial factors, Delta Airlines can implement preemptive measures, thus enhancing service quality and operational efficiency. Utilizing predictive analytics allows data-driven decisions that minimize disruptions caused by delays, leading to improved customer satisfaction and operational cost reduction.
# MAGIC
# MAGIC #####Operational and Financial Efficiency
# MAGIC Proactive delay management through predictive analytics translates to enhanced operational efficiency. Delta Airlines can optimize resource allocation based on predictive insights, resulting in smoother operations and reduced idle time for aircraft and personnel. Accurate delay predictions facilitate proactive measures, including optimized crew scheduling, reduced fuel wastage, and avoidance of penalties related to customer compensation for delays, thus reducing inefficiencies that lead to negative financial impact.
# MAGIC
# MAGIC #####Competitive Advantage
# MAGIC Timely performance remains a crucial competitive factor in the airline industry. Investing in predictive models to minimize delays positions Delta Airlines favorably in the market, appealing to passengers who prioritize reliability and punctuality.
# MAGIC
# MAGIC ###Usage
# MAGIC Our analysis enables Delta Airlines to proactively identify flights with high probabilities of dealy, allowing preemptive action to mitigate the damage caused by the delay. We recommend integrating this predictive model into operational strategies to optimize scheduling and mitigate delays. Continuous monitoring and refinement of the model promise sustained improvements in operational efficiency and customer satisfaction over time.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data and Feature Engineering
# MAGIC
# MAGIC Link to raw weather EDA & join: https://adb-4248444930383559.19.azuredatabricks.net/browse/folders/1128664627674437?o=4248444930383559<br>
# MAGIC Link to phase 1-2 exploration: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/3635549575866440/command/3635549575866441<br>
# MAGIC Link to cleaning code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319512363/command/1012234209193506<br>
# MAGIC Link to imputation code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319515426/command/1012234209193464<br>
# MAGIC Link to feature engineering code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319513866/command/1012234209193202

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Lineage and Transformations
# MAGIC The initial dataset in our project was the Reporting Carrier On-time Performance [1], which contains information for several years of U.S. flights including aircraft, carrier, airport, and delay timing information. In Phase 1 of this project, we began sifting through this dataset to understand its strengths and weaknesses. For example, we examined missing data, correlations among existing features, and both quantitative and research-based understandings of each of the recorded variables. The diagram below  depicts our overall work flow for the first half of our data pipeline. As can be seen from the diagram, we performied multiple iterations of EDA, feature selection, and feature engineering.  We will speak in depth to each of these steps in the following sections. 
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Data Pipeline Workflow</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/data_workflow.png?raw=true" width="400">
# MAGIC </div>
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ##### Joining Weather to OTPW Dataset
# MAGIC
# MAGIC One important note on the above pipeline is the join we performed to connect the weather and flights data. After some initial EDA, we decided to pursue the direction of rejoining the OTPW dataset, as this would allow us to work with many additional features that were otherwise incomplete or less precise in terms of timing. The datasets used for this join are illustrated below.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/prelim_data.png?raw=true" width="700">
# MAGIC </div>
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Below is a diagram depicting the logic used to re-join both the Daily and Hourly datasets to the OTPW dataset.  Due to the estimated nature of our UTC timestamps in the weather dataset, we created a new feature, 'three_hours_prior_to_departure' from which to join the weather observations. This additional hour buffer ensures that we do not violate the requirement of making predictions at least two hours prior to departure. 
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Weather Data Join Details</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/join.png?raw=true" width="700">
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC The daily weather features represent a 24-hour period summary of the given weather feature on the day prior to a flight.  Similarly, the hourly weather features can be interpreted as various weather observation windows up until approximately three hours before a flight. For more details on this join, please see our Phase II reporting notebook (link: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/3865911540703697/command/1012234209193366). We now move forward with the next phase of EDA and featuring engineering using this newly joined dataset. 
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ### Data Cleaning: Quantitative EDA
# MAGIC #### Dimensionality
# MAGIC After joining the data, we analyzed the integrity of the new dataset based on such dimensions as missing values and quantitative or conceptual redundancy in our features. One example of removing redundancy is the choice to drop many categorical variables, such as `ORIGIN_CITY_NAME`, `DEST_STATE_FIPS` and `DEST_STATE_NM` based on the rationale that this information was otherwise captured in retained flight-related variables such as `origin` and `destination` from which we could infer the same information. 
# MAGIC
# MAGIC In addition, we made the decision to reduce the dimensionality of our data by removing features with more than 80% nulls for their values. In addition, we removed observations such as cancelled flights that were not deemed representative of the population of flights we intended to make predictions for and/or did not have values for the target variable. 
# MAGIC
# MAGIC One important note is that after this initial dimensionality reduction on the entire dataset, we split off the 2019 data as a test set. We chose to do this prior to data imputation and feature engineering so that we would avoid leakage and "peeking" at the effects of our chosen imputations. See the section below (*Modeling and Leakage*) for more information. The table below summarizes the steps taken to reduce dimensionality and clean our dataset.
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <img src="https://github.com/baileykuehl/261_usage/blob/main/Screen Shot 2023-12-15 at 1.47.01 PM.png?raw=true" width="500">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC #### Imputations
# MAGIC After this initial cleaning step, we were left with features with very low percentages of nulls. However, these nulls needed to be addressedto prepare the data for the modeling stages. Key imputations for weather data include imputing missing values for snowfall and precipitation features as 0, to represent no weather of that type recorded for that time period. Based on the research that informed the design of the weather join, a null value for snowfall is likely to indicate no snowfall at that time, so nothing was recorded. For daily features, such as humidity, we felt it would be best to impute the nearest past day's value, given that certain weather features often are similar from day-to-day. We made these decisions to maximize our data available, but we acknowledge that this may have a minor effect on validity of our results. The descriptions of our imputation methods and the percentages of missing values by feature, both before and after imputation, are shown in the table below. 
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <img src="https://github.com/baileykuehl/261_usage/blob/main/Screen Shot 2023-12-15 at 11.05.53 AM.png?raw=true" width="500">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Feature Families and Visual EDA
# MAGIC We ultimately categorized our features into 3 main types: Weather Features, Flight Features, and Engineered Features. First, we will describe examples of the weather and flight feature families, as well as some initial visual EDA after reviewing the rejoined dataset. Many of the weather features became available after our join, as described above. The sub-categories of each weather and flight family are shown in the tables below. 
# MAGIC
# MAGIC ### 1. Weather Feature Family
# MAGIC We performed EDA on both the Daily and Hourly Datasets.  With our research findings indicating that commercial flights are resilient to weather, we chose to be aggressive in our preliminary feature reduction. For both the Daily and Hourly datasets, we kept only metric features. More complex features, such as "sky condition", would have required additional parsing logic.  We made the educated assumption that sky condition would not provide enough additional insight, beyond the metric features, to justify the additional time and compute resources necessary to include. 
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/weather_features.png?raw=true" width="800">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC After compiling groups of similar features, we calculated the Pearson Correlation Coefficient by subgroup accross the entire train dataset (2015-2018), and produced a pair plot for a small sampling of each. Below is an example of the calculated coefficients and sample pairplots for the metric temperature features found in the Hourly weather data.
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Weather Feature EDA Example: Hourly Temperature (F) Features Pairplot</strong>
# MAGIC     <br>
# MAGIC     3,000 Random Sample from (2015-2019)
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/pairplot.png?raw=true" width="500">
# MAGIC </div>
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC **Pearson Correlation Calculations for Hourly Temperature Features** *(Accross entire train dataset.)* 
# MAGIC |Feature Pair|Pearson Correlation Coeff.|
# MAGIC |---|---|
# MAGIC |'HourlyDewPointTemperature', 'HourlyDryBulbTemperature'|0.8313|
# MAGIC |'HourlyDewPointTemperature', 'HourlyWetBulbTemperature'|0.8671|
# MAGIC |'HourlyDryBulbTemperature', 'HourlyWetBulbTemperature'|0.8733|
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ### 2. Flight Feature Family
# MAGIC As our next category of feature families, we examined flight-related features. The flights dataset is originally sourced from the TranStats data collection (U.S. Department of Transportation). The full dataset consisted of on-time performance data for 31,746,841 U.S. passenger flights between 2015 and 2021, with 109 total features. However, as mentioned in sections above, we opted to remove features that were redundant. After narrowing down these columns and those with a significant number of missing datapoints, our remaining flight features are shown in the table below.
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC         <img src="https://github.com/baileykuehl/261_usage/blob/main/Screen Shot 2023-12-16 at 3.25.37 PM.png?raw=true" width="750">
# MAGIC </div>
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Below is an example of exploratory analysis we conducted on the flight data. While we were not surprised to encounter differences in average delays by airline (carrier), the level of variation across carriers helps to underscore the importance of carrier-based features in our models. We use this information in our modeling, as detailed in sections below.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC <strong>Flight Feature EDA Example: Average Departure Delay By Carrier</strong>
# MAGIC     <br>
# MAGIC     <br>
# MAGIC         <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/average_dep_delay_by_carrier.png?raw=true" width="700">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC
# MAGIC ### 3. Feature Engineering Family
# MAGIC Lastly, we also engineered a variety of useful features. One category of engineered features is related to seasonality. For example, we have `day_of_year` which assigns an integer value to the calendar date (1 - 365). Similarly, we have captured `is_holiday_window`, which is a binary value if the flight date falls within a window of 2 days before or 2 days after a federal holiday. We have chosen this window as many flights do not actually occur on the actual holiday, but in the days surrounding it.
# MAGIC
# MAGIC Another category of engineered features includes existing delay status at the time of prediction, e.g. `av_airport_delay` and `av_carrier_delay`. These two features provide us information about the airport-level and carrier-level delay attributes over a window of time preceeding the flight at hand. We expected these features to be effective predictors of future delays. In the table below are the categories and descriptions of our engineered features.
# MAGIC <br>
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <img src="https://github.com/baileykuehl/261_usage/blob/main/Screen Shot 2023-12-16 at 3.50.54 PM.png?raw=true" width="800">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC As an example of EDA with our engineered features, we present a heatmap (left) and pairplot (right). The heatmap shows us that there is a strong correlation between `airport_average_hourly` and `hourly_flights_origin`. This tells us that these two features are relatively similar, thus indicating we should likely choose one moving forward with our modeling. We also a see a relatively high correlation between `hourly_flights_origin` and `airport_congestion`, which is to be expected given that a high number of hourly flights at the origin airport would correspond to a large amount of congestion at the aiport.
# MAGIC
# MAGIC In the pair plot, we see a positive linear relationship between `departure_delay` and `prev_delay_tailnum`. Logically, this is to be expected, given that an initial delay of a single aircraft (tail number) likely cascades and causes delays for the rest of the day with that same aircraft. Another observation is that the average flight delay seems to be normally distributed with a slight right skew for the weekly number of flights per aircraft (`weekly_flights_tailnum`). 
# MAGIC
# MAGIC ##### Engineered Feature EDA Examples: Feature Heatmap and Pair Plots 
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/phase3_FE_heatmap.png?raw=true" width="500">       <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/phase3_FE_pairplot.png?raw=true" width="500">
# MAGIC </div>
# MAGIC <br>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### EDA Time Analysis
# MAGIC
# MAGIC Exploratory data analysis and transformations took a significant amount of our project time. The table below is meant to summarize the tasks and effort allotted prior to modeling.
# MAGIC
# MAGIC EDA Task|Total Estimated Time
# MAGIC  --|--
# MAGIC  Initial Analysis and Exploration<br>(feature understanding, redundancy evaluation, join logic) |7 days
# MAGIC  Data Join (compute time) |2.5 days
# MAGIC  Visualization Creation |1.5 days
# MAGIC  Nulls (analysis, compute) |1.5 hours 
# MAGIC  Imputations (analysis, compute) |3 hours 
# MAGIC  Feature Engineering (analysis, research) |6 hours 
# MAGIC  Feature Engineering (compute) |2 hours 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling and Leakage
# MAGIC
# MAGIC Link to LR code:https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1871383191021758/command/1012234209193386<br>
# MAGIC Link to RF code: https://adb-4248444930383559.19.azuredatabricks.net/browse/folders/1128664627674437?o=4248444930383559<br>
# MAGIC Link to MLP code:https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319515443<br>
# MAGIC Link to ensemble code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319532038/command/1012234209193150

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Modeling Pipeline and Checkpoints
# MAGIC
# MAGIC Our modeling pipeline began after all exploratory data analysis was performed. After adding final data imputations and engineered features to our dataset (excluding the held-out test set), we began building our models. The first step of modeling involved splitting the data into folds in order to avoid leakage. We designated the data from the year 2019 as a "held-out" test set not to be used until all model training and experimentation was completed and a final best model selected. To further avoid data leakage, we performed cross-validation due to the time-series nature of our data. 
# MAGIC
# MAGIC Next, we began building our predictive models. In the first stages of this project, we used a statistical baseline model by computing the average departure delay at the departure location and use this as proxy for delay of a given flight. Also in the early stages, we built a simple logistic regression which included only a shortlist of key features. As a preliminary step in this final phase of the project, we extended our Logistic regression model to include engineered features. Later, we develop new and more complex machine learning and AI models, includinf Random Forest, Multilayer Perceptron, and Ensemble models (please see the *Results and Discussion* section for model configuration and performance details).
# MAGIC
# MAGIC After we built our models, we experimented by training different variations of each model on our 4-year cross-validated training set, such as by including different subsets of features (e.g., based on feature importance evaluations), modifying modeling parameters (e.g., numTrees for random forest), and testing regularization techniques. This step was fairly iterative so that we could see the effects from each experiment and work collaboratively towards increasing our performance metrics. Note that each experiment was evaluated on both the training and validation sets at this point, but the held-out set mentioned earlier remained unevaluateed. Please see section below (*Subpipeline: Train, Test, Split and Timeline*) for more details on our data timeline for training and evaluating.
# MAGIC
# MAGIC Finally, we selected the models that yielded the best performances during experimentation as our "best model" for each model type. This was done through both grid search and checkpointing. Checkpointing, which is represented in our diagram with a blue checkpoint flag in our diagram, indicates stages at which we cached our data and/or saved the intermediate states of the model during the training process. These checkpoints were essential because they allowed us to experiment with different hyperparameters and model configurations in a timely manner. In the end, we were able to compare performance of each model across different checkpoints and identify which changes lead to improved performance (in terms of both speed and precision/recall) and easily revert to a better model.
# MAGIC
# MAGIC The full modeling pipeline is shown in the image below. 
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC <strong>Modeling Pipeline Workflow</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/baileykuehl/261_usage/blob/main/Screen Shot 2023-12-15 at 10.45.07 AM.png?raw=true" width="600">
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC
# MAGIC #### Subpipeline: Train, Test, Split and Timeline
# MAGIC
# MAGIC An important callout of our modeling pipeline is the train / test split performed, as well as our cross-validation strategy.  As depicted below, step 1 was splitting our dataset into a training dataset for 2015-2018 and a 2019 'holdout' set only to be used once for evaluation after our models were finalized. With our training dataset, we performed another split (step 2) which segregated 20% of the train dataset as a pure validation set.  We set this dataset aside as well and limited our usage of this for evaluation.   This new reduced train set was used for the bulk of our modeling and cross validation.  For all models, we used a time-series acceptable cross-validation technique (step 3) to ensure robustness of our model and prevent data leakage during the process.  (We will further discuss the technique and data leakage in the next section.)  Once we had built confidence in our cross-validated models, we then evaluated them against our pure validation dataset (step 4).  In this step we explored training various time periods with our model and also an ensemble of those results.  The theory behind this technique was to allow different fits to potentially capture the different high and low frequency features that existed in our dataset.  As an example, weather is extremely seasonal and can vary greatly from season to season.  We theorized that training on the entire dataset would give us the best chance for learning from weather features.  Other features, such as a specific airport, may portray higher frequency behavior.  As an example, an airport under construction several years ago, may have lead to higher delays for that airport during that time.  If this construction has since been completed, and delays caused by it have subsided, our model would likely perform better not training to that time period.  Unfortunately, we saw neglible differences in the various timeframes, and an ensemble of them did not see improvements either.  We speculate that the timeframes were not different enough to truly capture meaningful differences while also providing enough data to properly train. For that reason, we chose to scrap this technique and instead select a single timeframe for each model. 
# MAGIC
# MAGIC After iterations through the cross-validation and pure validation sets, we selected our final model pipeline and evaluated it against the test holdout dataset.  We will discuss the specific model and the results in further sections. 
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC <strong>Cross-Val and Train/Test Split</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/train_val_test.png?raw=true" width="600">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC ### Leakage
# MAGIC
# MAGIC Data leakage occurs when information from an evaluation dataset (data the model should not have access to during training) "leaks" into the training process.  This leakage can lead to heightened evaluation results, and a false sense of a models true performance.  Future models using the same data will likely underperform.  Leakage can occur at multiple sets and in various ways. Below we highlight some of the ways that leakage could occur and how we designed our pipeline to mitigate it, as best as possible. 
# MAGIC
# MAGIC - **EDA:** We were careful to ensure that no stages of our EDA included the 2019 dataset.  This ensured that we were not learning from, or building logic based on, that specific time period.  All decisions made for feature reduction, imputations, feature engineering and transformation were all made based of our findings from this EDA. We then transformed the test dataset accordingly. While this kept our final test holdout set pure, it does introduce leakage to our interim steps like the pure validation. In theory a new EDA could have been performed for each stage of training to ensure that there was no leakage for a specific stage.  However, practially this introduces a lot of additional compute, rebuild of engineered features and transformation and would likely result in marginal benefit. 
# MAGIC
# MAGIC - **Features from the Future:** Our flight dataset contained several features with observations made within two hours of a flight's departure, or in some cases occured after a flight departed.  Some examples of such features are "Wheels Off" *(the amount of time it took to get the wheels off the runway of the flight at hand)* and "Taxi Out' *(time taken to taxi to the runway)*. These features are happening just before the flight in question takes off, and would not be available at the "prediction time" 2 hours before the flight. We deligently reviewed all features and the accompanying documentation to ensure that none of these features we utilized directly in modelling.   This same methodalogy applied to the engineered features that we created as well. In some instances we created features that were derived from previous flights records (i.e. "average flight delay by carrier), but for these features we carefully selected observation windows that ensured that we did not violate the two hours prior to departure buffer. 
# MAGIC
# MAGIC - **Time-Series Cross-Validation Technique:** As indicated previously, we utilized a cross-validation technique specific for minimizing potential data leakage with time-series data.  With standard cross-validation, a dataset is broken up into folds and iterative training is performed using each fold as an evaluation set. In our instance this would lead to using future events to predict the past, which violates our purpose. To negate this, we used what is referred to as a sliding time-series split.  As depicted in the split diagram in the previous section, we separated our train into three equal, but overlapping, folds.  The train set for each fold butt up to each other, to ensure that we do not have gaps in what we are training.  The test sets are then chronologically after each train and overlap with the next train's set (except for the last fold).  This ensures that there is no time-based data leakage in training the folds.  
# MAGIC
# MAGIC - **Layers of Evaluation:** There will inevitably be data leakage in the initial layers of model training as we iterate through cross-validation models to tweak feature selection and hyperparameter tuning.  There may also be some leakage in the fact that each test set from a cross-validation fold overlaps with the training of another.  However, we have designed our pipeline with layers of evaluation to mitigate the effect data leakage in the initial steps may have on the production performance of our model.  The pure validation set was used sparsely and the test evaluation only once.  These layers diminish any performance surprises from data leakage for this model when applied to new data. 
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ### Loss Functions
# MAGIC The models we used during this phase include Logistic Regression, Random Forest, MLP, and an ensemble with all 3 models. For logistic regression, we utilize cross entropy, also known as the log (logarithmic) loss, as our loss function. This loss function is chosen because it quantifies the difference between predicted probabilities and actual values, making it an important classification metric [4]. 
# MAGIC
# MAGIC For our Random Forest (RF) model, we know that RF doesn't explicitly optimize a loss function like other gradient-based methods(e.g. logistic regression). However, Spark's RF classifier uses *Gini impurity* as the default impurity measure for splitting nodes during tree construction. In this context, Gini impurity is used to evaluate the impurity of a node. It quantifies the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled based on the class labels in the node [8]. This equation is defined below:
# MAGIC
# MAGIC $${Gini} = 1 - \sum_{i=1}^{C} p_i^2$$
# MAGIC
# MAGIC Finally, our Neural Network (MLP) has a loss function, Cross Entropy, which is used by our model as part of the optimization process rather than explicitly as a parameter (as it would in PyTorch or Tensorflow). Cross Entropy Loss is optimized using optimization techniques (e.g. stochastic gradient descent) during training.
# MAGIC
# MAGIC $$Cross Entropy Loss = \-\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]\$$
# MAGIC
# MAGIC Where:
# MAGIC - N is the number of samples
# MAGIC - yi is the true label (0 or 1)
# MAGIC - pi is the predicted probability for class 1
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results and Discussion
# MAGIC Notebooks to produce the graphs<br>
# MAGIC Validation data: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319525821/command/1012234209193114<br>
# MAGIC Test Data: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1012234209200373/command/1012234209200383

# COMMAND ----------

# MAGIC %md
# MAGIC ### Results
# MAGIC
# MAGIC As mentioned throughout this report, we have opted to begin demonstrating this problem through our statistical baseline, as well as a simple random guess. The purpose of these baselines is to demonstrate improvements we can make over simple prediction strategies. The improvements involved machine learning and AI models, including logistic regression, random forest, multi-layer perceptrons, and ensemble. We summarize all modeling experiments and report their best performances in the table below.
# MAGIC <br>
# MAGIC
# MAGIC ##### Results Summary: Comparative performance of models
# MAGIC  Model|Wall time (min) | Hyperparameters    | Feature Family (count) | Num Input Features| Training Dataset | Validation Precision (%)| Train Precision (%)| Computational Config.
# MAGIC  --|--|--|--|--|--|--|--|--
# MAGIC  Random Guess| 1.5|-|-|-| 1 Year |18.4 |- |16 GB Memory<br>4 cores
# MAGIC  Baseline| 1.5|- |Avg flight delay for airport<br>Avg over -4 to -2 hrs|2| 4 Years |23.8 |23.8|16 GB Memory<br>4 cores
# MAGIC  Trivial LR| 18 |5 epochs <br> regparam = 0.005 <br> elasticNetParam = 1|Numerical features (26) <br>Categorical features (5)| 31 |4 Years |19.6|19.7 |28 GB Memory<br>8 cores
# MAGIC  Engineered LR| 39|5 epochs <br> regparam = 0.005 <br> elasticNetParam = 1 |Trivial LR Features (31)<br>Engineered features (9)|40| 4 Years |26.5 |27.0| 28 GB Memory<br>8 cores
# MAGIC  Random Forest| 95 |numTrees = 50 <br> maxDepth = 30 <br>inInstancesPerNode=1000<br>minInfoGain = 0.0001|Engineered weather (18)<br>Engineered time/airline/airport (8) <br>Raw time/airline/airport (6)|32| 4 Years |27 |27.2|28 GB Memory<br>8 cores
# MAGIC  Multilayer Perceptron| 8.5 |20 Numeric features<br>hidden layers = 2 x 4<br>maxIter  = 50<br>tol = E-6<br>step = 0.03| Numeric features with moderate-to high DT importance (20 )|20 | 4 Years |25.5 |25.5|28 GB Memory<br>8 cores
# MAGIC  Ensemble|+20 min |Majority voting with weighted votes |Features from LR + RF+ MLP|40| 4 Years |26.5 |26.5 |28 GB Memory<br>8 cores
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ### Discussion of Results
# MAGIC
# MAGIC The graphs below exhibit performance of the models on the validation (left) and test (right) data sets.
# MAGIC
# MAGIC To build these charts, we analyzed predicted probability of the positive label, produced by the individual models. Varying threshold for calling a flight delayed or not we can djust precision and recall to the desired levels. FOr instance, if the model predicted the probability of a delay to be 40%, and our decision threshold is 30%, we label the flight as delayed. If, however, our decision threshold is 50%, we label the same flight as delayed. This way we directly demonstrate usefullness of the model to the stakeholders, something that might be unclear if we were to use F1 or F2 metrics. 
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC <strong>Model performance on validation (left) and test (right) datasets</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/Models_on_val.jpg?raw=true" width="500">        <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/Models_on_test.jpg?raw=true" width="500">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC The table below summarizes our key performance metric of precision at 80% recall.
# MAGIC
# MAGIC
# MAGIC  Model|Precision at 80% recall (%) | | 
# MAGIC  --|--|--|
# MAGIC --| Validation set|Held-out test set|
# MAGIC  Random Guess|18.4|18.3|
# MAGIC  Baseline|23.8|23.1|
# MAGIC  MLP|25.5|26.4|
# MAGIC  Log Regression|26.5|26.5|
# MAGIC  Random Forest|27|26.9|
# MAGIC
# MAGIC ##### Random Guess
# MAGIC Random guess, the simples possible model, yields the same ~18% precision at all elvels of the recall. That means, that we predict the flight delayed with some fixed probability, then the precision does not depend on the probability, but the recall does.
# MAGIC
# MAGIC ##### Non-ML baseline
# MAGIC Baseline model, which is just the average delay at the origin airport known at the time of prediction, already  perfoms a lot better, showing almost 30% relative improvement over random guess. It is interesting to notice, that 80% recall is a high bar for our models. Have we measured the performance at 50% recall, performance difference would be a lot more pronounced. However, we belive that only catching every other delayed flight will not be actionbale for our client.
# MAGIC
# MAGIC ##### Logistic regression
# MAGIC Logistic regression demonstrates some improvement over MLP, likely becasue it is not as compute instensive. Interestingly, in our preliminary experiments, LR only trained on the features diretcly avaliable in the dataset yielded only low performance. Introduction of the features based on the known delays dramatically improved its predictive power. We suggest that this is becasue features in the data set lack substantial predictive power. In other words, they do not matter as much. There are other factors, such as aircraft maintanance and the airline mamangement practicies that strongly affect delays. These features are not present in the dataset, but we get a glimpse of them inderectly, through observed delays. That is what introduction of delay-based features improves model performance so much.
# MAGIC
# MAGIC ##### Random Forest 
# MAGIC The top performing model (only slightly better than the logistic regression) is the Random Forest. This is likely because of its ability to capture non-linear patterns with little compute power. Since our target metric was to maximize precision while holding recall at 80%, we had to get creative with how to accomplish this with a random forest model. Instead of extracting the models predictions, we instead extracted the ratios of on-time vs. delayed flights for each leaf node. We translated this ratio of labels as the predicted probability of delay. A record was then given the 'probability' of the node for which it fell into. For hyper-parameter tuning we tested a grid search during our cross-validation stages of modeling. Below is a table showing the parameters of the grid search, as well as a table for the best performing parameter combinations for each fold. 
# MAGIC
# MAGIC **Hyperparameters for Grid Search**
# MAGIC
# MAGIC | Hyper Parameter | List of Values |
# MAGIC |-----------------| -------------- |
# MAGIC | 'maxDepth'      |  [5, 10, 15]   |
# MAGIC | 'numTrees'      |  [50, 75, 100] |
# MAGIC | 'minInfoGain'   |  [0.01, 0.001, 0.0001] |
# MAGIC
# MAGIC **Best Performing Combinations**
# MAGIC
# MAGIC | Fold | Parameters |
# MAGIC |-----| -------------- |
# MAGIC | 1   |  'maxDepth': 15 <br> 'numTrees': 50 <br> 'minInfoGain': 0.0001 |
# MAGIC | 2   | 'maxDepth': 15 <br> 'numTrees': 100 <br> 'minInfoGain': 0.0001 |
# MAGIC | 3   |  'maxDepth': 15 <br> 'numTrees': 50 <br> 'minInfoGain': 0.0001 |
# MAGIC
# MAGIC
# MAGIC We only ran the grid search once, and for a couple different reasons. First, due to the expensive nature of grid searches, we wanted to limit our usage unless it was providing significant value. Second, time contrainsts limited our options to pre-made grid search modules available for pyspark. This created two issues for us: 1) The grid search we found available requires accompanying validation folds, and those folds do not meet time-series requirements. 2) The uniqueness of our selected metric (precision at held 80% recall) is not one that is readily available as an evaluator of the grid search.  For the grid search we settled with AUC as the evaluator.  For these reasons we kept the results of the grid search in mind, but carried forward with more hueristic manual tuning of the RF parameters.  With more time, we would have opted to develop a custom grid search that addresses these two concerns.  
# MAGIC
# MAGIC With the random forest model performing similar to the logistic regression, reinforces the notion that the data set lacks features with predictive power.
# MAGIC
# MAGIC ##### Multilayer Perciptron
# MAGIC To develop our multilayer perceptron model, we selected the subset of numeric features with at least moderate importance values across multiple rounds of the decision tree modeling and logistic regression as input. Following data preprocessing, we experimented with the four distinct network architectures detailed in the table below. 
# MAGIC
# MAGIC | MLP model | Architecture | Avg. CV Time |  Avg. CV Train Recall  |  Avg. CV Train Recall  | Avg. CV Train Recall |  Avg. CV Train Recall  | 
# MAGIC | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ |
# MAGIC |1 | 20 - 8 - Relu - 2 Softmax  | 166.02s  | 0.723 |  0.358 | 0.615 |  0.578|
# MAGIC |2 | 20 - 4 - Relu - 2 Softmax   | 196.53s  | 0.723 | 0.358 | 0.615 | 0.566 |
# MAGIC |3 | 20 - 4 - Relu - 4 - Relu - 2 Softmax  |  244.06s | 0.711 | 0.598 | 0.617| 0.599 | 
# MAGIC |4 | 20 - 16 - Relu - 8 - Relu - 4 - Relu - 2 Softmax  | 296.09s | 0.723 |  0.356 | 0.630 |0.582 |
# MAGIC
# MAGIC Ultimately, modifications to the network architectures translated only to minimal differences in average precision and recall values across the cross-validated sets. Architecture 3 (20 - 4 - Relu - 4 - Relu - 2 Softmax) was selected as our final MLP model on the basis of its marginally lower disparities between precision and recall for the validation sets against the train sets in cross validation as well as its lighter compute time compared to the more complex Architecture 4 (20 - 16 - Relu - 8 - Relu - 4 - Relu - 2 Softmax).
# MAGIC There are a number of possible explanations for the somewhat static performance of the MLP, regardless its architecture: among them, it's possible that using a wider range of input features would have yielded more improvements. However, it's also possible that an alternative model such as a convolutional neural network would be more suitable to analyzing this type of time-series data [14].
# MAGIC
# MAGIC ### Best Pipeline and Gap Analysis
# MAGIC Trying to find synergy in performance of the three models, we compiled the top models into an ensemble. To do that we calculated weighted average probability, and then adjusted the final threshold on this average probability to achieve 80% recall. We did not observe any improvement with this approach, likely becasue the all models rely on the same patterns in the data.
# MAGIC
# MAGIC The metrics for ensamble performance on various datasets is summarized in te table below. The models were re-trained on the fianal train dataset (2015-2018) before transforming the held-out dataset (2019). The ensemble parameters used in the transformation of the test dataset were selected from the experimets on the train data set and were not adjusted anymore to avoid leakage. Consistent performance of the models accross various datasets points to minimal overfitting and robust performance of all three models. 
# MAGIC
# MAGIC Metric|Validation|Train|Test
# MAGIC ---|---|---|---
# MAGIC Precison|26.5|26.5|26.5
# MAGIC Recall|79.5|79.9|80.9
# MAGIC F1|41|39.8|57.8
# MAGIC F2|57.8|57.0|57.3
# MAGIC
# MAGIC The graph below shows distribution of the delays separately for the subset that was predicted by the model as as "delayed" and "on time". 
# MAGIC Importantly, longer delays have higher chance of being predicted as delayed. This demonstrates that the model does find some patterns predictive of the delay. In fact, for flights with the delay significantly above 15, most are labeled as delayed. Therefore, we hope that changing the delay threshold from 15 min to more reasonable 30 min will help the performance of our model. On the other hand, many flights that depart ahead of schedule are also labeled as "delayed" by the model. This might be an indication that the model finds flights that are unusual in general, rather than flights that are specifically delayed. We hope that re-engineering the features with the delays below 15 min set to 0, might remedy this issue. 
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC <strong>Delay distribution among flights predicted to be delayed and on-time by the final model</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/Label_distribution.jpg?raw=true" width="400">
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC
# MAGIC In summary, the purpose of this project is to use machine learning to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. This work has important implications for the general public because flight delays are extremely disruptive for both commercial and leisure passengers, flight crew, and other airline employees. We hypothesized that, using Random Forest and MLP models, we could achieve a precise, reliable methodology for predicting flight delays.  We contributed additional features through careful engineering, some of the most impactful features for modeling included average delay at the origin airport, average airline carrier delay, flights occurring in a holiday window, and more. We also contributed a custom-joined dataset containing weather and airline features, which expands upon the available data from the original OTPW dataset. Finally, we have contributed proof of concept towards our hypothesis by building successful Random Forest and MLP models. Our top modeling pipeline included our ensemble model, which achieved a maximum precision of 26.5% at a recall of 80.9%. The various models feeding the ensemble, their hyperparameters, and how they were integrated are depicted below. (*For features used, please refer to the previously provided model table*)
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC <strong>Ensemble Model Pipeline</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/ensemble.png?raw=true" width="400">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC The results from this pipeline are signficant because they get us one step closer to understanding how delays occur and provide instrumental knowledge for Delta Airlines. We hope that Delta will use our key predictive features to their advantage and start making changes to mitigate delays.
# MAGIC
# MAGIC ### Future Directions
# MAGIC Though our time on this project has come to and end, we feel that there is plenty of additional work to be done if our colleagues choose to pick this project up at a later date. One simple addition to our work would be to include categorical variables in the MLP model. Another future direction would be to re-engineer some of our features, such as imputing negative delays as 0’s, as we found that leaving them in affected our model's performance. Additionally, joining external datasets to the existing data could close the gap on current missing features, which include maintenance logs for aircrafts. Finally, we would recommend the next team to tackle this problem reconsider the delay threshold of 15 minutes. From our analysis, we found that 15 minute delays are quite difficult to predict, and might be inconsequential to customers compared to delays greater than 30 minutes. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## References
# MAGIC Inline citations throughout the report are represented as bracketed references, e.g. *[4]*.     
# MAGIC <br>
# MAGIC
# MAGIC 1. On Time Performance and Weather (OTPW) Dataset, original source: https://www.transtats.bts.gov/homepage.asp  
# MAGIC Analysis of the Influence of Factors on Flight Delays in the United States Using the Construction of a Mathematical Model and Regression Analysis: https://ieeexplore.ieee.org/document/9795721
# MAGIC 2. Time Series and Cross Validation: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20
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
# MAGIC 14. Del Pra, M. (2020, September 8). Time Series Classification with Deep Learning. https://towardsdatascience.com/time-series-classification-with-deep-learning-d238f0147d6f
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: 
# MAGIC
# MAGIC ### Gantt Chart & Credit Assignment Table
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
# MAGIC
# MAGIC <br>
# MAGIC <html>
# MAGIC   <head>
# MAGIC   </head>
# MAGIC   <body>
# MAGIC     <div style="text-align: center;">
# MAGIC       <img src="https://github.com/esambrailo/w261/blob/main/delta.jpg?raw=true" width="600">
# MAGIC    </div>
# MAGIC   </body>
# MAGIC </html>
# MAGIC <br>
