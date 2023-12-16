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
# MAGIC Our project aims to leverage machine learning techniques to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. The goal of this work is to give Delta airlines a tool to flag a flight as a potentially delayed so that they can better anticipate potential delays and mitigate them. To narrow down our dataset to key predictive features, we performed extensive exploratory data analysis, rejoined the *On Time Performance and Weather (OTPW)* dataset [3], and used feature selection via setting a null threshold and Lasso Regularization. We built, trained, and evaluated logsitic regression, random forest, and multilayer perceptron (MLP) models. Finally, we experimented with combining these three models to create an ensemble model. To measure model performance, we chose to evaluate precision at a threshold of 80% recall, which we belive would be the minimum recall of an actianble prediction. We threshold our predictions such that the resulting recall is approximately 80% and strive to achive the highest precision possible at this threshold. Our best model is a Random Forest that demosntrtraed 29% precision on the test dataset. Our best modeling pipeline included engineered features (e.g. average delay at the origin airport, hourly precipitatation, etc.) and grid search for hyperparameter tuning. We hope to use our predictive model as a part of Delta's workflow to help them mitigate potential losses before they can occur.
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
# MAGIC Delta Airlines is a prominent airline carrier, who, from time-to-time grapples with challenges related to flight delays impacting customer satisfaction and operational efficiency. To help Delta tackle this issue, our Data Science team conducted a thorough analysis using machine learning techniques to predict delays exceeding 15 minutes within 2 hours of scheduled departure. Our aim is to furnish Delta Airlines with actionable insights, enabling proactive mitigation of the damage casued by the delay and disruption reduction. Our technical approach involved comprehensive data exploration, feature engineering, and the utilization of diverse machine learning models (logistic regression, random forest, and multilayer perceptron) to achieve a precise prediction of flight delays. Our methodology stands as a reliable tool for predicting and managing these delays effectively.
# MAGIC
# MAGIC ###Impact
# MAGIC #####Customer Satisfaction
# MAGIC Our foremost goal is to leverage historical flight data and weather information to construct predictive models accurately anticipating delays. By discerning crucial factors, Delta Airlines can implement preemptive measures, thus enhancing service quality and operational efficiency. Utilizing predictive analytics allows data-driven decisions that minimize disruptions caused by delays, leading to improved customer satisfaction and operational cost reduction.
# MAGIC
# MAGIC #####Operational and Financial Efficiency
# MAGIC Proactive delay management through predictive analytics translates to enhanced operational efficiency. Delta Airlines can optimize resource allocation based on predictive insights, resulting in smoother operations and reduced idle time for aircraft and personnel. Accurate delay predictions facilitate proactive measures, including crew schedule optimization, reduced fuel wastage, and avoidance of penalties related to customer compensation for delays, thus reducing inefficiencies that lead to negative financial impact.
# MAGIC
# MAGIC #####Competitive Advantage
# MAGIC Timely performance remains a crucial competitive factor in the airline industry. Investing in predictive models to minimize delays positions Delta Airlines favorably in the market, appealing to passengers who prioritize reliability and punctuality.
# MAGIC
# MAGIC ###Usage
# MAGIC Our analysis enables Delta Airlines to proactievely identify flights with high probability of dealy, allowing preemptive action to mitigate the damage caused by the delay. We recommend integrating this predictive model into operational strategies to optimize scheduling and mitigate delays. Continuous monitoring and refinement of the model promise sustained improvements in operational efficiency and customer satisfaction over time.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data and Feature Engineering
# MAGIC
# MAGIC Link to cleaning code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319512363/command/1012234209193506<br>
# MAGIC Link to imputation code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319515426/command/1012234209193464<br>
# MAGIC Link to feature engineering code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319513866/command/1012234209193202

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Lineage and Transformations
# MAGIC The dataset that we started with was the Reporting Carrier On-time Performance [3] which contains flight data (including weather, airport / aircraft information, etc.). In Phase 1 of this project, we began sifting through this dataset to understand its strengths and weaknesses. For example, we looked for missing data, correlations among existing features, and attempted to understand the meanings behind each of the columns provided. The below diagram depicts our overal work flow for the first half of our data pipeline. As can be seen from the diagram, we are performing multiple iterations of EDA, feature selection, and feature engineering.  We will speak in depth to each of these steps in the following sections. 
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
# MAGIC One important callout from the above pipeline is the join, which connects the weather and flight data. After some initial EDA, we decided to pursue the direction of rejoining the OTPW dataset, as this would allow us to work with many additional features. A conceptual diagram of this join is shown below.
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
# MAGIC Below is a diagram depicting the logic that was used for joining both the Daily and Hourly datasets to the OTPW dataset. *(wb5)*  Due to the estimated nature of our UTC timestamps in the weather dataset, we created a new feature, 'three_hours_prior_to_departure' from which to join the weather observations. *(wb5)* This additional hour buffer ensures that we do not violate the requirement of making predictions two hours prior to departure. 
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
# MAGIC The daily weather data can be interpretted as a summary of the weather the day before the flight.  The hourly weather features can be interpreted as various weather observation windows up until approximately three hours before a flight. We now move forward with the next phase of EDA and featuring engineering using this newly joined dataset. 
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ### Data Cleaning: Quantitative EDA
# MAGIC #### Dimensionality
# MAGIC After joining the data, we analyzed the new dataset for potential nulls, missing values, redundancy in our features, etc. We made the decision to reduce the dimensionality of our data by removing features with more than 80% nulls for their values. In addition, we removed rows that did not have the target variable available. The table below summarizes the steps taken to reduce dimensionality and clean our dataset. 
# MAGIC
# MAGIC One important note is that after this initial dimensionality reduction on the entire dataset, we split off the 2019 data as a test set. We chose to do this prior to data imputation and feature engineering so that we would avoid leakage and "peeking" at the effects of our chosen imputations. See the section below (*Modeling and Leakage*) for more information.
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <img src="https://github.com/baileykuehl/261_usage/blob/main/Screen Shot 2023-12-15 at 1.47.01 PM.png?raw=true" width="500">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC #### Imputations
# MAGIC After this initial cleaning step, we were left with features that had a much smaller percentage of nulls. However, these nulls needed to be addressed so that the dataset was ready to be used in the modeling stages. Key imputations for weather data include imputing missing values as 0's. After researching the weather information from the resulting join, we felt 0 was the most appropriate value, as there was likely nothing recorded as a result of the lack of that feature. For example, a null for snowfall likely indicates there was no snowfall at that time, so nothing was recorded. For daily features, such as humidity, we felt it would be best to impute the nearest day's value, given that certain weather features often are similar from day-to-day. We made these decisions to maximize our data available, but we acknoweldge that this may affect the performance of our models. The descriptions of our imputations and the null percentages before and after are shown in the table below. 
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
# MAGIC We have categorized our features into 3 main groups: Weather Features, Flight Features, and Engineered Features. First, we will show examples of the weather and flight feature families, as well as some initial visual EDA after reviewing the rejoined dataset. Many of the weather features became available after our join, as described above. The sub-categories of each weather and flight family are shown in the tables below. 
# MAGIC
# MAGIC ### 1. Weather Feature Family
# MAGIC We performed EDA on both the Daily and Hourly Datasets.  With our research findings indicating that commercial flights are resilient to weather, we chose to be aggressive in our preliminary feature reduction.  For both the Daily and Hourly datasets, we kept only metric features *(wb3&4)*. More complex features, such as "sky condition", would have required additional parsing logic.  We made the educated assumption that sky condition would not provide enough additional insight, beyond the metric features, to justify the additional time and compute resources necessary to include. 
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/weather_features.png?raw=true" width="800">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC We compiled groups of similar features.  For each grouping, we calculated the Pearson Correlation Coefficient accross the entire train dataset (2015-2018), and produced a pair plot for a small sampling of each. Below is an example of the calculated coefficients and sample pairplots for the metric temperature features found in the Hourly weather data. *(The results of all grouped features can be found in the notebooks (nb3&4))*
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC     <strong>Weather Feature EDA Example: Hourly Temperature (F) Features Pairplot</strong>
# MAGIC     <br>
# MAGIC     3,000 Random Sample from (2015-2019)
# MAGIC     <br>
# MAGIC     <img src="https://github.com/esambrailo/w261/blob/main/pairplot.png?raw=true" width="700">
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
# MAGIC As our next category of feature families, we looked at flight-related features. As demonstrated from our dimensionali
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC         <img src="https://github.com/esambrailo/w261/blob/13f7a831ab4e77b37423acbfe83375262fe5609b/flight_data.png?raw=true" width="550">
# MAGIC </div>
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC While we were not surprised to encounter differences in average delays by airline (carrier), the level of variation across carriers helps to underscore the importance of carrier-based features in our models. 
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC <strong>Flight Feature EDA Example: Average Departure Delay By Carrier</strong>
# MAGIC     <br>
# MAGIC     <br>
# MAGIC         <img src="https://github.com/ArtemChemist/team_5_3/blob/main/Images/average_dep_delay_by_carrier.png?raw=true" width="800">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC
# MAGIC ### 3. Feature Engineering Family [LUCY - ADD VISUALIZATIONS]
# MAGIC Lastly, we also engineered a variety of features. One category of engineered features is related to seasonality. For example, we have `day_of_year` which assigns an integer value to the calendar date (1 - 365). Similarly, we have captured `is_holiday_window`, which is a binary value if the flight date falls within a window of 2 days before or 2 days after a federal holiday. We have chosen this window as many flights do not actually occur on the actual holiday, but in the days surrounding it.
# MAGIC
# MAGIC Another category of engineered features includes existing delays, e.g. `av_airport_delay` and `av_carrier_delay`. These two features aim to give us information about the airport and the carrier delays over a window of time preceeding the flight at hand. We expected these features to be good predictors of future delays.
# MAGIC
# MAGIC Below are the categories and descriptions of our engineered features.
# MAGIC <br>
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC     <br>
# MAGIC     <img src="https://github.com/baileykuehl/261_usage/blob/main/Screen Shot 2023-12-14 at 7.55.45 PM.png?raw=true" width="400">
# MAGIC </div>

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
# MAGIC Link to RF code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1012234209200548/command/1012234209200549<br>
# MAGIC Link to MLP code:https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/3215205203386644/command/1012234209193268<br>
# MAGIC Link to ensemble code: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319532038/command/1012234209193150

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Modeling Pipeline and Checkpoints
# MAGIC
# MAGIC For this project, our modeling pipeline began after all exploratory data analysis was performed. After adding final data imputations and engineered features to our dataset (excluding the held-out test set), we began building our models. The first step of modeling involved splitting the data into folds in order to avoid leakage. We designated the data from the year 2019 as a "held-out" test set which we will not used until all model training and experimentation is completed and a final best model is selected. To further avoid data leakage, we performed cross-validation given that our data is time series. Please see 
# MAGIC
# MAGIC Next, we began building our models. In the first stages of this project, we used a statistical baseline model by computing the average departure delay at the departure location and use this as proxy for delay of a given flight. Also in the early stages, we built a simple logistic regression which included only a shortlist of features. As a preliminary step in this final phase, we built upon our Logistic regression model to include engineered features. Later, we added on to our list of models and built more complex, machine learning and AI models, which included Random Forest, Multilayer Perceptron, and Ensemble models (please see the *Results and Discussion* section for model configuration and performance details).
# MAGIC
# MAGIC After we built our models, we experimented by training different variations of each model on our 4-year cross-validated training set, such as including different subsets of features (based on feature importance evaluations), modifying modeling parameters (e.g. numTrees for random forest), and testing regularization techniques. This step was fairly iterative so that we could see the effects from each experiment and work towards increasing our performance metrics. Note that each experiment was evaluated on both the training and validation sets at this point, but the held-out set mentioned earlier has yet to be evaluated on. Please see section below (*Subpipeline: Train, Test, Split and Timeline*) for more details on our data timeline for training and evaluating.
# MAGIC
# MAGIC Finally, we selected the models that resulted in the best performances during experimentation as our "best model" for each experiment. This was done through both grid search and checkpointing. Checkpointing, which is represented in our diagram with a blue checkpoint flag in our diagram, indicates places where we cached our data and/or saved the intermediate states of the model during the training process. These checkpoints are important because they allowed us to experiment with different hyperparameters and model configurations in a timely manner. In the end, we were able to compare performance of each model across different checkpoints and identify which changes lead to improved performance (in terms of both speed and precision/recall) and easily revert to a better model.
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
# MAGIC An important callout of our modeling pipeline is the train / test split performed, as well as our cross-validation strategy. The procedure for and benefits of this cross-validation are explained in the section below (*Leakage*).
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC <strong>Cross-Val and Train/Test Split</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/baileykuehl/261_usage/blob/main/image (4).png?raw=true" width="600">
# MAGIC </div>
# MAGIC <br>
# MAGIC
# MAGIC ### Leakage
# MAGIC
# MAGIC Data leakage occurs when information from the test set (data the model should not have access to during training) "leaks" into the training process. For example, a leak could arise from including features in the training set that would not be available at the time of prediction or using information from the test set to preprocess training data can lead to inflated performance metrics and misleading model evaluations. 
# MAGIC
# MAGIC **-- Are you violating any cardinal sins of ML?**
# MAGIC
# MAGIC **-- Describe how your pipeline does not suffer from any leakage problem and does not violate any cardinal sins of ML**
# MAGIC
# MAGIC The main way that we avoided leakage in our pipeline was to perform cross-validation. As can be seen from the above figure (*Cross-Val and Train/Test Split*), we split off the 2019 data as a test set immediately after initial dimensionality reduction on our dataset. We chose to do this prior to data imputation and feature engineering so that we would avoid leakage and "peeking" at the effects of our chosen imputations. In addition, we **[!!!!!! details about cross-validation]**
# MAGIC
# MAGIC Another step that we took to avoid leakage in our data was to remove any features that would not have been available at the actual time of prediction. For example, we are trying to predict flights that would be delayed *within 2 hours of scheduled departure.* Yet, some of the features provided in the dataset included wheels off (the amount of time it took to get the wheels off the runway of the flight at hand) and taxi out (time taken to taxi to the runway). These features are happening just before the flight in question takes off, and would not in fact be available at "prediction time" which would be 2+ hours before the flight. We conciously removed these features to avoid this kind of data leakage.
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experimental Results Summary: Comparative performance of models
# MAGIC
# MAGIC We summarize all model experiments and their best performances in the table below.
# MAGIC
# MAGIC  Model|Wall time (min) | Hyperparameters    | Feature Family (count) | Num Input Features| Training Dataset | Test Precision (%)| Train Precision (%)| Computational Config.
# MAGIC  --|--|--|--|--|--|--|--|--
# MAGIC  Random Guess| 1.5|-|-|-| 1 Year |18.4 |- |16 GB Memory<br>4 cores
# MAGIC  Baseline| 1.5|- |Avg flight delay for airport<br>Avg over -4 to -2 hrs|2| 4 Years | |23.8|16 GB Memory<br>4 cores
# MAGIC  Trivial LR| 18 |5 epochs <br> regparam = 0.005 <br> elasticNetParam = 1|Numerical features (26) <br>Categorical features (5)| 31 |4 Years | |19.7 |28 GB Memory<br>8 cores
# MAGIC  Engineered LR| 39|5 epochs <br> regparam = 0.005 <br> elasticNetParam = 1 |Trivial LR Features (31)<br>Engineered features (9)|40| 4 Years | |27.0| 28 GB Memory<br>8 cores
# MAGIC  Random Forest| 120 |numTrees = 100 <br> maxBins = 200<br>maxDepth = 30 <br>inInstancesPerNode=1000<br>minInfoGain = 0.0001|Weather (26)<br>Engineered features (9)|35| 4 Years | |27.5|28 GB Memory<br>8 cores
# MAGIC  Multilayer Perceptron| 125 |18 Numeric features<br>hidden layers = 2 x 28<br>maxIter  = 20<br>tol = E-6<br>step = 0.03|Features selected by engineered LR (37)|37| 4 Years | |25.5|28 GB Memory<br>8 cores
# MAGIC  Ensemble|+20 min |Majority voting with weighted votes |Features from LR + RF+ MLP|40| 4 Years | | |28 GB Memory<br>8 cores
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ### Discussion of Results
# MAGIC - discuss models, experiments: successes and surprises
# MAGIC <br>
# MAGIC <div style="text-align: center;">
# MAGIC <strong>Modeling Pipeline Workflow</strong>
# MAGIC     <br>
# MAGIC     <img src="https://github.com/baileykuehl/261_usage/blob/main/Models_on_val.jpg?raw=true" width="600">
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC #### Random Forest
# MAGIC - grid search
# MAGIC
# MAGIC #### Neural Network (MLP)
# MAGIC
# MAGIC -- Implement Neural Network (NN) model
# MAGIC
# MAGIC -- Experiment with at least 2 different Network architectures and report results.
# MAGIC
# MAGIC -- Report neural network architecture in string form (e.g., 100 - 200 - Relu - 300 - Relu - 2 Softmax )
# MAGIC
# MAGIC
# MAGIC #### Ensemble
# MAGIC
# MAGIC ### Best Pipeline and Gap Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC
# MAGIC In summary, the purpose of this project is to use machine learning to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. This work has important implications for the general public because flight delays are extremely disruptive for both commercial and leisure passengers, flight crew, and other airline employees. We hypothesized that, using Random Forest and MLP models, we could achieve a precise, reliable methodology for predicting flight delays.  We contributed additional features through careful engineering, some of the most impactful features for modeling included average delay at the origin airport, average airline carrier delay, flights occurring in a holiday window, and more. We also contributed a custom-joined dataset containing weather and airline features, which expands upon the available data from the original OTPW dataset. Finally, we have contributed proof of concept towards our hypothesis by building successful Random Forest and MLP models. **Our top modeling pipeline included our xxx model, which achieved a maximum precision of xxx% at a recall of 80%.** The features and hyperparameters for this model (obtained through grid search) are shown below. 
# MAGIC
# MAGIC **[insert summary table]**
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
# MAGIC ## Appendix: Gantt Chart & Credit Assignment Table
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
