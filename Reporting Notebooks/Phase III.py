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
# MAGIC Our project aims to leverage machine learning techniques to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. The goal of this work is to advise Delta airlines on the key factors that influence delays so that they can better anticipate potential delays and mitigate them. For this project, we sought to narrow down our dataset to key predictive features through extensive exploratory data analysis and feature engineering. We also aimed to build and train logsitic regression, random forest, and multilayer perceptron (MLP) models. To measure our model performances, we chose to evaluate precision at a threshold of 80% recall, selected for the minimum recall required to be accepted in industry. As such, our baseline model and best logistic regression model resulted in precision values of 24.8% and 31%, respectively. As for our best random forest classifier, chosen for its ability to specify feature importance, we achieved a precision of 28%. Thus, our best modeling pipeline was ???, which involved adding engineered features including average delay at the origin airport and engineered weather features, and iterating on our models through additional feature engineering and grid search. We hope to use our insights about key factors affecting airline delays to inform our employer, Delta Airlines, and help them mitigate potential causes for delays before they can occur.
# MAGIC
# MAGIC ### Phase 2 Abstract
# MAGIC Our project aims to leverage machine learning techniques to predict whether an airline flight will be delayed by 15+ minutes within 2 hours of its scheduled departure. The goal of this work is to advise Delta airlines on the key factors that influence delays so that they can better anticipate potential delays and mitigate them. Similarly to Phase 1, we utilized approximately 75% of the data in the 1-year dataset from the *On Time Performance and Weather (OTPW)* dataset -- an airline and weather dataset containing data from 2015 - 2019 [1]. During Phase 2 of our project, we sought to narrow down our dataset to key predictive features through extensive exploratory data analysis and feature engineering. Additionally, we aimed to train a baseline model on our data by predicting the average delay, which we believe to be a fitting baseline because it is simple in terms of computation and resources, but has room for improvement through more advanced models. Also during this phase, we sought to build and train more complicated models, including logsitic regression and random forest. As described during Phase 1, we have chosen to measure our model performances in terms of precision while maintaining our desired 80% recall, selected for the minimum recall required to be accepted in industry. As such, our baseline model and best logistic regression model resulted in precision values of 24.8% and 31%, respectively. As for our best random forest classifier, chosen for its ability to specify feature importance, we achieved a precision of 28%. Thus, our best modeling pipeline was from our experimentation with logistic regression, which involved adding engineered features including average delay at the origin airport and engineered weather features. For the next and final phase of this project, we hope to iterate on our current models to further improve performance values. Such iterations might include additional feature engineering (such as adding an isHolidayWindow feature), potentially joining an additional dataset to our current data, and fine-tuning existing model parameters through grid search. We hope to optimize our model in order to gain insights about key factors affecting airline delays so that we can share our results with our employer, Delta Airlines, and help them mitigate potential causes for delays before they can occur.
# MAGIC
# MAGIC ### Phase 1 Abstract
# MAGIC Air travel in the United States is the preferred method of transportation for commercial and recreational use because of its speed, comfort, and safety [1]. Given its initial popularity, air travel technology has improved significantly since the first flight took off in 1908 [2]. For example, modern forecasting technology allows pilots to predict the optimal route and potential flight delays and cancellations given forecasted headwinds, storms, or other semi-predictable events. However, previous studies have found that weather is actually not the primary indicator of whether a flight will be delayed or canceled [1]. Today, seemingly unexpected flight delays are not only a nuisance for passengers, but also could a potentially detrimental threat to the airline industry if customers continue to lose trust in public airline capabilities. Thus, the primary goal of this project is to predict flights delays more than 15 minutes in duration that occur within 2 hours prior to the expected departure time. To accomplish this, we will extract airline and weather data spanning the years 2015 - 2019 from the *On Time Performance and Weather (OTPW)* dataset [3]. Feature selection will be performed through null thresholding (dropping features with more than 90% nulls) and lasso regularization. Key features are hypothesized to be Airline (e.g. *Delta, Southwest*), expected maintenence, history of delays for a given flight number (route), and extreme weather (e.g. ice or snow) [4]. We will perform data cleaning, imputation, and exploratory analysis on the remaining data. The cleaned data will be split into test, train, and validation sets via cross-validation on a rolling basis given the time series nature of the data. We will then build and train a logisitic regression model as a baseline, as well as a random forest to predict delays. The proposed pipeline is expected to perform efficiently in terms of runtime given our proposed utilization of partitionable parquet files in place of the more commonly used CSV files. Finally, to measure the success of our model, we propose to use precision and recall, optimizing the tradeoff between the two such that precision is maximized given a goal recall of 80%. Using the results from this project, we hope to advise airlines on key factors affecting flight delays so that they can mitigate them to better satisfy their customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data and Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC -- Summarize the data lineage and key data transformations (joins)
# MAGIC
# MAGIC -- List of feature families explored and explanation of each
# MAGIC
# MAGIC -- List of features within each family and description of each, along with THEIR EDA
# MAGIC
# MAGIC -- Please refer to experiments showing the value of each feature/family
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Neural Network (MLP)
# MAGIC
# MAGIC -- Implement Neural Network (NN) model
# MAGIC
# MAGIC -- Experiment with at least 2 different Network architectures and report results.
# MAGIC
# MAGIC -- Report neural network architecture in string form (e.g., 100 - 200 - Relu - 300 - Relu - 2 Softmax )
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Modeling Pipelines
# MAGIC
# MAGIC -- A visualization of the modeling pipeline (s) and subpipelines if necessary
# MAGIC
# MAGIC -- Families of input features and count per family
# MAGIC
# MAGIC -- Number of input features
# MAGIC
# MAGIC -- Hyperparameters and settings considered
# MAGIC
# MAGIC -- Loss function used (data loss and regularization parts) in latex
# MAGIC
# MAGIC -- Number of experiments conducted
# MAGIC
# MAGIC -- Experiment table with the following details per experiment:
# MAGIC
# MAGIC   - Baseline experiment
# MAGIC
# MAGIC   - Any additional experiments
# MAGIC
# MAGIC   - Final model tuned
# MAGIC
# MAGIC   - best results (1 to 3) for all experiments you conducted with the following details
# MAGIC
# MAGIC   - Computational configuration used
# MAGIC
# MAGIC   - Wall time for each experiment
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Leakage
# MAGIC
# MAGIC -- Define what is leakage and provide a a hypothetical example of leakage
# MAGIC
# MAGIC -- Go through your Pipeline and check if there is any leakage.
# MAGIC
# MAGIC -- Are you violating any cardinal sins of ML?
# MAGIC
# MAGIC -- Describe how your pipeline does not suffer from any leakage problem and does not violate any cardinal sins of ML
# MAGIC
