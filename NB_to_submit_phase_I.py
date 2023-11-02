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
# MAGIC |Bailey Kuehl|Week 1|
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
# MAGIC ||2|Art||
# MAGIC ||2|Erik||
# MAGIC ||2|Lucy||
# MAGIC ||2|Bailey||
# MAGIC ||3|Art||
# MAGIC ||3|Erik||
# MAGIC ||3|Lucy||
# MAGIC ||3|Bailey||
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
# MAGIC
# MAGIC #### Airlines    
# MAGIC [Lucy]
# MAGIC
# MAGIC
# MAGIC #### Weather 
# MAGIC [Erik]
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Data
# MAGIC
# MAGIC #### Quantitative (mean, std)
# MAGIC [Bailey]
# MAGIC
# MAGIC #### Visuals
# MAGIC Historic trend of delays
# MAGIC
# MAGIC Historic trend of precepitation
# MAGIC
# MAGIC Historic trend of delays at one airport

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
# MAGIC #### Challenges
# MAGIC - Seasonality of data
# MAGIC - COVID period at the end of data
# MAGIC - Efficiency of operations on large data
# MAGIC - Missing data
# MAGIC
# MAGIC #### Conclusions 
# MAGIC - TBD
# MAGIC
# MAGIC #### Next Steps
# MAGIC - In depth ETA on joined dataset
# MAGIC - Feature selection and engineering
# MAGIC - Generating a baseline
# MAGIC
