# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: Multilayer Perceptron Model 
# MAGIC
# MAGIC **Team 5-3: Bailey Kuehl, Lucy Herr, Artem Lebedev, Erik Sambraillo** 
# MAGIC <br>**261 Fall 2023**
# MAGIC <br>
# MAGIC <br>
# MAGIC This notebook contains our implementation of a multi-layer perceptron (MLP) using the OTPW dataset to predict flight delays of 15 minutes or more at 2 hours prior to scheduled departure. This version of the data has been re-joined from the original flights and weather tables.
# MAGIC <br>
# MAGIC <br>
# MAGIC **Overview of process**
# MAGIC - Data preprocessing for MLP
# MAGIC - Experiment with network architectures
# MAGIC     - Architecture 1: 20 - 8 - Relu - 2 Softmax 
# MAGIC     - Architecture 2: 20 - 4 - Relu - 2 Softmax 
# MAGIC     - Architecture 3: 20 - 4 - Relu - 4 - Relu - 2 Softmax 
# MAGIC     - Architecture 4: 20 - 16 - Relu - 8 - Relu - 4 - Relu 2 Softmax
# MAGIC - Evaluate chosen model against "pure" train and validation sets
# MAGIC - Evaluate chosen model against complete train and test (holdout) sets 
# MAGIC - Evaluate unbalanced train set with chosen model for overfitting analysis
# MAGIC
# MAGIC
# MAGIC **Sources**
# MAGIC - https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier
# MAGIC - Full documentation: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.MultilayerPerceptronClassifier.html
# MAGIC - MLP for time series data : https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/
# MAGIC

# COMMAND ----------

# DBTITLE 1,Import modules
# custom functions for project
import funcs

# summary data/plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data pre-processing - encoding & feature transformation
from pyspark.sql import Window,  DataFrame
from pyspark.sql.functions import col,  mean, lag, coalesce, stddev, min, max, count, when, isnan, percent_rank, year, month, udf, array, sum as _sum
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler # Tokenizer, HashingTF,
from pyspark.sql.types import NumericType,StringType,DoubleType,ArrayType, IntegerType

# modeling
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier, MultilayerPerceptronClassificationSummary
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time 


# COMMAND ----------

# DBTITLE 1,Connect to team blob storage
# connect to team blob
team_blob_url = funcs.blob_connect()
# view blob storage - cleaned data directory
display(dbutils.fs.ls(f"{team_blob_url}/BK/"))

# COMMAND ----------

# DBTITLE 1,Load clean 5-year OTPW data
# load clean 5-year OTPW data with engineered features (train set = 2015-2018) 
df = spark.read.parquet(f"{team_blob_url}/BK/clean_5yr_WITHOUT_2019_eng/")

# COMMAND ----------

# DBTITLE 1,Define multilayer perceptron modeling functions 
def mlp_model_cv(df_tuple, class_weight, mlp_pipeline, verbose=True):
    '''
    Function that builds and fits a multilayer perceptron model. 
    Model downsamples dataset for balanced binary label, using class distribution.
    
    Inputs:
     - 'df_tuple': Expects to receive train and validation dfs as a tuple: (train, val). 
     - 'class_weight': percentage of positive binary label.
     - 'dt_model: the initialized model with desired params

    Ouput:
    Returns a dataframe of both validation and train results with 
    columns: ['probability', 'prediction', 'label'].
    '''
    start = time.time()

    # extracting train & validation from tuple
    train = df_tuple[0]
    val = df_tuple[1]

    #quantifying fraction for downsampling on-time flights
    fraction = class_weight/(1-class_weight)

    if verbose:
            print("Downsampling . . .")

    # downsampling on-time flights
    on_time_train = train[train['DEP_DEL15'] == 0].sample(fraction=fraction)
    # temp collection of delayed
    delayed_train = train[train['DEP_DEL15'] == 1]
    # recreating downsampled train df
    dwnsmpl_train = on_time_train.union(delayed_train)

    if verbose:
        print("Fitting model . . .")

    # fit model 
    fitted_model = mlp_pipeline.fit(dwnsmpl_train)

    # train/val predictions 
    train_results = fitted_model.transform(dwnsmpl_train)
    val_results = fitted_model.transform(val)
    
    # collect results for evaluation metrics 
    result_cols = ['probability', 'prediction', 'label']
    train_results = train_results.select(result_cols)
    validation_results = val_results.select(result_cols)

    if verbose:
        print("Results complete.")
    
    # print elapsed time 
    elapsed = time.time() - start
    print(f"Modeling time: {elapsed}")

    return train_results, validation_results


def mlp_model_res(df_tuple, class_weight, mlp_pipeline, verbose=True):
    '''
    Function that builds and fits a multilayer perceptron model. 
    Model downsamples dataset for balanced binary label, using class distribution.
    
    Inputs:
     - 'df_tuple': Expects to receive train and validation dfs as a tuple: (train, val). 
     - 'class_weight': percentage of positive binary label.
     - 'dt_model: the initialized model with desired params

    Ouput:
    Returns a dataframe of both validation and train results with 
    columns: ['probability', 'prediction', 'label'].
    '''
    start = time.time()

    # extracting train & validation from tuple
    train = df_tuple[0]
    val = df_tuple[1]

    #quantifying fraction for downsampling on-time flights
    fraction = class_weight/(1-class_weight)

    if verbose:
            print("Downsampling . . .")

    # downsampling on-time flights
    on_time_train = train[train['DEP_DEL15'] == 0].sample(fraction=fraction)
    # temp collection of delayed
    delayed_train = train[train['DEP_DEL15'] == 1]
    # recreating downsampled train df
    dwnsmpl_train = on_time_train.union(delayed_train)

    if verbose:
        print("Fitting model . . .")

    # fit model 
    fitted_model = mlp_pipeline.fit(dwnsmpl_train)

    # train/val predictions 
    train_results = fitted_model.transform(dwnsmpl_train)
    val_results = fitted_model.transform(val)
    
    # collect results for evaluation metrics 
    result_cols = ['sched_depart_date_time_UTC','TAIL_NUM','probability', 'prediction', 'label']
    train_results = train_results.select(result_cols)
    validation_results = val_results.select(result_cols)

    if verbose:
        print("Results complete.")
    
    # print elapsed time 
    elapsed = time.time() - start
    print(f"Modeling time: {elapsed}")

    return train_results, validation_results

def evaluate_cv_mlp_and_print_metrics(train_res_list,val_res_list,metrics = ['precisionByLabel','recallByLabel']):
    '''
    Input paired lists of cross-validation train and validation result dataframes for MLP model.
    Default metrics are 'precisionByLabel' and 'recallByLabel' by delayed class.
    '''
    for i, (train_results, val_results) in enumerate(zip(lst_train_results, lst_val_results)):
        print(f"Results for Dataset #{i}")
        for metric in metrics:
            evaluator.setMetricName(metric).setParams(metricLabel=1) # precision by delayed class
            train_metric = evaluator.evaluate(train_results)
            val_metric = evaluator.evaluate(val_results)
            print(f"Train {metric}: {train_metric}")
            print(f"Validation {metric}: {val_metric}")
            print("-" * 70)

def evaluate_full_mlp_and_print_metrics(train_res_list,val_res_list,metrics=['precisionByLabel','recallByLabel']):
    '''
    Input paired train and validation/test result dataframes for MLP model.
    Default metrics are 'precisionByLabel' and 'recallByLabel' by delayed class.
    '''
    for metric in metrics:
        evaluator.setMetricName(metric).setParams(metricLabel=1) # precision by delayed class
        train_metric = evaluator.evaluate(train_results)
        val_metric = evaluator.evaluate(val_results)
        print(f"Train {metric}: {train_metric}")
        print(f"Validation/test {metric}: {val_metric}")
        print("-" * 70)


# COMMAND ----------

# DBTITLE 1,Additional team modeling functions (Erik/Artem)
def cross_val_percentages(num_blocks=5, split_ratio=0.8):
    '''
    Creates cross validation block percentiles for both the train and validation sets
    based off the number of blocks and split ratios identified.
    '''
    # creating percentile boundaries for train and validation blocks
    val_area = 1- (1-split_ratio) * 1/num_blocks
    train_block = (1-split_ratio) * 1/num_blocks
    train_blocks_boundaries = [(val_area*i/num_blocks, val_area*(i+1)/num_blocks) for i in range(num_blocks)]
    val_blocks_boundaries = [(val_block[1], val_block[1] + train_block ) for val_block in train_blocks_boundaries]
    print("Train blocks: ", train_blocks_boundaries)
    print("Validation blocks: ", val_blocks_boundaries)
    return train_blocks_boundaries, val_blocks_boundaries

def create_validation_blocks(df, split_feature, blocks=5, split=0.8):
    '''
    Function that orders and ranks a df based on a specified feature, 
    and then splits it into equal train and validation blocks based off
    the specified number of blocks and split percent.
    Returns a list of tuples for the train and validation datasets.
    '''
    # defining the window feature for splitting
    window_spec = Window.partitionBy().orderBy(split_feature)

    # creating a rank column for ordered df
    ranked_df = df.withColumn("rank", percent_rank().over(window_spec))
    
    # creating cross validation percentiles
    train_blocks, val_blocks = cross_val_percentages(blocks, split)

    # Assemble tuples of train and val datasets for cross-validations
    val_train_sets = []
    for train_b, val_b in zip(train_blocks, val_blocks):
        val_train_sets.append((
                                ranked_df.where(f"rank <= {train_b[1]} and rank >= {train_b[0]}").drop('rank')
                                , ranked_df.where(f"rank > {val_b[0]} and rank <= {val_b[1]}").drop('rank')
                                ))
    return val_train_sets

def extract_prob(v):
    '''Convert probability output column to a column with probability of positive.'''
    try:
        return float(v[1])
    except ValueError:
        return None
    
def combining_results(list):
    '''
    Combining a list of dataframes into a single dataframe.
    '''
    results = list[0]
    for result in list[1:]:
        results = results.union(result)
    return results

def TP(prob_pos, label):
    '''Returning an array of 1's for all True Positives by Cutoff'''
    return [ 1 if (prob_pos >= cut_off) and (label == 1)  else 0 for cut_off in CutOffs]

def FP(prob_pos, label):
    '''Returning an array of 1's for all False Positives by Cutoff'''
    return [ 1 if (prob_pos >= cut_off) and (label == 0)  else 0 for cut_off in CutOffs]

def FP_TP_dataframe(df, CutOffs=[0, 0.30, 0.40, 0.45, 0.60, 0.80]):
    '''
    Function to label a prediction as FP(TP), based on various cutoffs, 
    and map these arrays to df as new columns.
    Expects a dataframe with columns: 'probability', 'label'
    CutOffs to be a list of float percentages.
    '''
    # extracting the delayed flight probabilities from results
    extract_prob_udf = udf(extract_prob, DoubleType())
    df = df.withColumn("prob_pos", extract_prob_udf(col("probability")))

    # Define udfs based on these functions
    # These udfs return arrays of the same length as the cut-off array
    # With 1 if the decision would be TP(FP) at this cut off
    make_TP = udf(TP,  ArrayType(IntegerType()))
    make_FP = udf(FP,  ArrayType(IntegerType()))

    # Generate these arrays in the dataframe returned by prediction
    prediction = df.withColumns({'TP':make_TP(df.prob_pos, df.label), 'FP':make_FP(df.prob_pos, df.label)})

    # Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
    num_cols = len(CutOffs)
    TP_FP_pd = prediction.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                            array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                            _sum(col("label")).alias("Positives")
                            )\
                            .toPandas()
    # Convert the result into the pd df of precisions and recalls for each cut-off
    results_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
    results_pd['Precision'] = 100*results_pd['TP']/(results_pd['TP'] + results_pd['FP'])
    results_pd['Recall']= 100*results_pd['TP']/TP_FP_pd.iloc[0,2]
    return results_pd

def plot_precision_recall(train_table, val_table):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_table.Recall, train_table.Precision, '-o', label='Train')
    ax.plot(val_table.Recall, val_table.Precision, '-o', label='Validation')
    ax.legend(fontsize=8)
    plt.figtext(0.5, 0.92, "Precision-Recall Performance:\n precision-recall tradeoff at various probability thresholds", ha="center", va="center", fontsize=10)

    ax.set_ylim(10, 50)
    ax.set_xlim(70, 90)
    ax.set_xlabel('Recall (%)', size=10)
    ax.set_ylabel('Precision (%)', size=10)

    # Write cutoff vaulues on the graph
    for index in range(len(train_table.Cutoff)):
        ax.text(train_table.Recall[index]-0.02, 1 + train_table.Precision[index], train_table.Cutoff[index], size=9)
        ax.text(val_table.Recall[index]-0.02, 1 + val_table.Precision[index], val_table.Cutoff[index], size=9)

    # Draw a vertical line to show 80% recall
    ax.axvline(x=80, ymin=0, ymax=60, color='gray', ls = '--')
    # ax.text(68, 84, '80% Recall', size=8)

    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data pre-processing for MLP
# MAGIC - select features based on decision tree model feature importance 
# MAGIC - cross validation splits
# MAGIC - address class imbalance 
# MAGIC - vectorize
# MAGIC - standardize features

# COMMAND ----------

# DBTITLE 1,Feature selection
# load list of important feature values from "50-tree grove"
feat_import_dfs = spark.read.parquet(f"{team_blob_url}/LH/feature_importance_50treegrove")
# list important features
important_features = [row['feat_name'] for row in feat_import_dfs.select("feat_name").distinct().collect()]

# select important features that weren't one-hot encoded in DT
non_ohe_feats = [item for item in important_features if not any(substring in item for substring in ["DEST", "ORIGIN", "OP_UNIQUE_CARRIER", "origin_type"])]

# list original features for OHE versions on important list
orig_version_ohe_feats = ['OP_UNIQUE_CARRIER','ORIGIN','DEST','origin_type']
all_important_features = non_ohe_feats + orig_version_ohe_feats

# list numeric features with higher DT importance
all_numeric_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, NumericType)]

# intersection of numeric and immportant features
numeric_important_features = list(set(all_numeric_columns).intersection(all_important_features))

# select unique ID columns (flight date + plane tail number), target, and important numeric features for modeling
id_cols = ["sched_depart_date_time_UTC","TAIL_NUM"]

# select model-only features in 5-year data 
df = df.select(*numeric_important_features, *id_cols, "DEP_DEL15")#col("DEP_DEL15").alias("label"))

# COMMAND ----------

# DBTITLE 1,Generate cross-validation sets
# generate cross-validation sets -  unique key: flight date + tail number

# create pure validation and train sets
pure_val_tuples = create_validation_blocks(df, "sched_depart_date_time_UTC", blocks=1, split=0.8)

# create cross-validations sets
cross_val_tuples = create_validation_blocks(pure_val_tuples[0][0], "sched_depart_date_time_UTC", blocks=3)

#compute label distribution for each cross validation set
class_weights = []
for i, (train, val) in enumerate(cross_val_tuples):
    # developing class weights
    total = train.count()
    delayed = train[train['DEP_DEL15'] == 1].count()
    percent = delayed/total

    class_weights.append(percent)
print("Class Weights: ", class_weights)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment with network architectures

# COMMAND ----------

# DBTITLE 1,Architecture 1: [20, 8, 2]
# build pipeline 1
layers = [20,8,2]
assembler = VectorAssembler().setInputCols(numeric_important_features).setOutputCol('feat_vec')
scaler = StandardScaler().setInputCol('feat_vec').setOutputCol('feat_scaled')
lbl_indexer = StringIndexer(stringOrderType = 'alphabetAsc').setInputCol('DEP_DEL15').setOutputCol('label')
classifier = MultilayerPerceptronClassifier(featuresCol="feat_scaled",
                                            labelCol="label",
                                            maxIter=50,
                                            layers=layers)

pipeline1 = Pipeline(stages=[assembler, scaler,lbl_indexer,classifier])

# initiating lists for cross-validation results
lst_train_results = []
lst_val_results  = []

# iterate through each cross-val tuple, run model, & append results to list 
for i, df_tuple in enumerate(cross_val_tuples):
    print("-"*70)
    print(f"Starting dataset #{i}")
    train_results, validation_results = mlp_model(df_tuple, class_weights[i], pipeline1, verbose=True)
    lst_train_results.append(train_results)
    lst_val_results.append(validation_results)

# COMMAND ----------

# evaluate cross-fold validation results for pipeline 1 and compute performance metrics 
evaluate_cv_mlp_and_print_metrics(lst_train_results,lst_val_results)

# COMMAND ----------

# DBTITLE 1, Architecture 2: [20, 4, 2]
# build pipeline  2
layers = [20,4,2]
assembler = VectorAssembler().setInputCols(numeric_important_features).setOutputCol('feat_vec')
scaler = StandardScaler().setInputCol('feat_vec').setOutputCol('feat_scaled')
lbl_indexer = StringIndexer(stringOrderType = 'alphabetAsc').setInputCol('DEP_DEL15').setOutputCol('label')
classifier = MultilayerPerceptronClassifier(featuresCol="feat_scaled",
                                            labelCol="label",
                                            maxIter=50,
                                            layers=layers)
pipeline2 = Pipeline(stages=[assembler, scaler,lbl_indexer,classifier])

# initiating lists for cross-validation results
lst_train_results = []
lst_val_results  = []

# iterate through each cross-val tuple, run model, & append results to list 
for i, df_tuple in enumerate(cross_val_tuples):
    print("-"*70)
    print(f"Starting dataset #{i}")
    train_results, validation_results = mlp_model(df_tuple, class_weights[i], pipeline2, verbose=True)
    lst_train_results.append(train_results)
    lst_val_results.append(validation_results)


# COMMAND ----------

# evaluate cross-fold validation results for pipeline 2 and compute performance metrics 
evaluate_cv_mlp_and_print_metrics(lst_train_results,lst_val_results)

# COMMAND ----------

# DBTITLE 1, Architecture 3: [20, 4, 4, 2]
# build pipeline 3 
layers = [20,4,4,2]
assembler = VectorAssembler().setInputCols(numeric_important_features).setOutputCol('feat_vec')
scaler = StandardScaler().setInputCol('feat_vec').setOutputCol('feat_scaled')
lbl_indexer = StringIndexer(stringOrderType = 'alphabetAsc').setInputCol('DEP_DEL15').setOutputCol('label')
classifier = MultilayerPerceptronClassifier(featuresCol="feat_scaled",
                                            labelCol="label",
                                            maxIter=50,
                                            layers=layers)
pipeline3 = Pipeline(stages=[assembler, scaler,lbl_indexer,classifier])

# initiating lists for cross-validation results
lst_train_results = []
lst_val_results  = []

# iterate through each cross-val tuple, run model, & append results to list 
for i, df_tuple in enumerate(cross_val_tuples):
    print("-"*70)
    print(f"Starting dataset #{i}")
    train_results, validation_results = mlp_model(df_tuple, class_weights[i], pipeline3, verbose=True)
    lst_train_results.append(train_results)
    lst_val_results.append(validation_results)


# COMMAND ----------

# evaluate cross-fold validation results for pipeline 3 and compute performance metrics 
evaluate_cv_mlp_and_print_metrics(lst_train_results,lst_val_results)

# COMMAND ----------

# DBTITLE 1,Architecture 4: [20, 16, 8, 4, 2]
# build pipeline 4
layers = [20,16,8,4,2]
assembler = VectorAssembler().setInputCols(numeric_important_features).setOutputCol('feat_vec')
scaler = StandardScaler().setInputCol('feat_vec').setOutputCol('feat_scaled')
lbl_indexer = StringIndexer(stringOrderType = 'alphabetAsc').setInputCol('DEP_DEL15').setOutputCol('label')
classifier = MultilayerPerceptronClassifier(featuresCol="feat_scaled",
                                            labelCol="label",
                                            maxIter=50,
                                            layers=layers)
pipeline4 = Pipeline(stages=[assembler, scaler,lbl_indexer,classifier])

# initiating lists for cross-validation results
lst_train_results = []
lst_val_results  = []

# iterate through each cross-val tuple, run model, & append results to list 
lst_train_results = []
lst_val_results = []
for i, df_tuple in enumerate(cross_val_tuples):
    print("-"*70)
    print(f"Starting dataset #{i}")
    train_results, validation_results = mlp_model(df_tuple, class_weights[i], pipeline4, verbose=True)
    lst_train_results.append(train_results)
    lst_val_results.append(validation_results)


# COMMAND ----------

# evaluate cross-fold validation results for pipeline 4 and compute performance metrics 
evaluate_cv_mlp_and_print_metrics(lst_train_results,lst_val_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate "pure" train & validation sets

# COMMAND ----------

pure_train = spark.read.parquet(f"{team_blob_url}/BK/pure_train/")
pure_val = spark.read.parquet(f"{team_blob_url}/BK/pure_val/")

# select unique ID columns (flight date + plane tail number), target, and important numeric features for modeling
id_cols = ["sched_depart_date_time_UTC","TAIL_NUM"]

# select model-only features in 5-year data 
pure_train = pure_train.select(*numeric_important_features, *id_cols, "DEP_DEL15")
pure_val = pure_val.select(*numeric_important_features, *id_cols, "DEP_DEL15")


# COMMAND ----------

# compute label distribution for each cross validation set
class_weights = []
for i, (train, val) in enumerate(pure_val_tuples):
    # developing class weights
    total = train.count()
    delayed = train[train['DEP_DEL15'] == 1].count()
    percent = delayed/total
    class_weights.append(percent)
print("Class Weights: ", class_weights)

# COMMAND ----------

# implement chosen mlp model to pure train & validation results 
pure_train_results, pure_validation_results = mlp_model_res(pure_val_tuple, class_weights[i], 
                                                            pipeline3, verbose=True)

# evaluate results for pure train/val and compute performance metrics 
for metric in metrics:
    evaluator.setMetricName(metric).setParams(metricLabel=1) # precision by delayed class
            train_metric = evaluator.evaluate(train_results)
            val_metric = evaluator.evaluate(val_results)
            print(f"Train {metric}: {train_metric}")
            print(f"Validation {metric}: {val_metric}")
            print("-" * 70)

# COMMAND ----------

# format column names for comparison across models 
pure_validation_results = pure_validation_results.withColumnRenamed("prediction", "mlp_pred_lbl")
pure_validation_results = pure_validation_results.withColumnRenamed("probability", "mlp_prob_pos")
# write results to storage
#pure_validation_results.write.mode('overwrite').parquet(f"{team_blob_url}/LH/MLP/mlp_final_val")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final test set evaluation (holdout set ~2019)

# COMMAND ----------

# load clean 5-year OTPW data with engineered features (train set = 2015-2018) 
final_train = spark.read.parquet(f"{team_blob_url}/BK/final_train/")
final_test = spark.read.parquet(f"{team_blob_url}/BK/final_test/")

# select unique ID columns (flight date + plane tail number), target, and important numeric features for modeling
id_cols = ["sched_depart_date_time_UTC","TAIL_NUM"]

# select model-only features in 5-year data 
final_train = final_train.select(*numeric_important_features, *id_cols, "DEP_DEL15")
final_test = final_test.select(*numeric_important_features, *id_cols, "DEP_DEL15")

test_tuple = final_train,final_test

# COMMAND ----------

train_results, test_results = mlp_model_res(test_tuple, class_weights[i], pipeline3, verbose=True)

metrics = ['precisionByLabel', 'recallByLabel']
for metric in metrics:
    evaluator=MulticlassClassificationEvaluator(metricName=metric).setParams(metricLabel=1) # delayed class
    train_metric = evaluator.evaluate(train_results)
    test_metric = evaluator.evaluate(test_results)
    print(f"Train {metric}: {train_metric}")
    print(f"Test {metric}: {val_metric}")

# COMMAND ----------

# format column names for comparison across models 
pure_test_results = test_results.withColumnRenamed("prediction", "mlp_pred_lbl")
pure_test_results = pure_test_results.withColumnRenamed("probability", "mlp_prob_pos")
# write results to storage
mlp_final_test.write.mode('overwrite').parquet(f"{team_blob_url}/LH/MLP/mlp_final_test")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform unbalanced raw 2015-1028 dataset with final model for overfitting evaluation

# COMMAND ----------

# transform unbalanced raw 2015-1028 dataset with final model: 

# load clean 5-year OTPW data with engineered features (train set = 2015-2018) 
final_train = spark.read.parquet(f"{team_blob_url}/BK/final_train/")
# select unique ID columns (flight date + plane tail number), target, and important numeric features for modeling
id_cols = ["sched_depart_date_time_UTC","TAIL_NUM"]
# select model-only features in 5-year data 
final_train = final_train.select(*numeric_important_features, *id_cols, "DEP_DEL15")

# compute class weight 
total = final_train.count()
delayed = final_train[final_train['DEP_DEL15'] == 1].count()
final_train_weight = delayed/total

# downsampling on-time flights
on_time_train = train[train['DEP_DEL15'] == 0].sample(fraction=final_train_weight)
# temp collection of delayed
delayed_train = train[train['DEP_DEL15'] == 1]
# recreating downsampled train df
dwnsmpl_train = on_time_train.union(delayed_train)

# fit model 
fitted_model = pipeline3.fit(dwnsmpl_train)

# train/val predictions 
unbalanced_train_results = fitted_model.transform(final_train) # UNBALANCED TRAIN SET 
    
# collect results for evaluation metrics 
result_cols = ['sched_depart_date_time_UTC','TAIL_NUM','probability', 'prediction', 'label']
unbalanced_train_results = unbalanced_train_results.select(result_cols)

# COMMAND ----------

# rename columns for comparison across models 
unbalanced_train_results = unbalanced_train_results.withColumnRenamed("prediction", "mlp_pred_lbl")
unbalanced_train_results = unbalanced_train_results.withColumnRenamed("probability", "mlp_prob_pos")
# write to storage
unbalanced_train_results.write.mode('overwrite').parquet(f"{team_blob_url}/LH/MLP/mlp_unbalanced_train")

# COMMAND ----------

# DBTITLE 1,Generate TP/FP cutoff dataframes  & plot
# union of all results into a single dataframe
all_train_results = combining_results(lst_train_results)
all_val_results = combining_results(lst_val_results)

# Convert probability output column to a column with probability of positive
extract_prob_udf = udf(lambda x: float(x[1]), DoubleType())
all_val_results = all_val_results.withColumn("prob_pos", extract_prob_udf(col("probability")))
all_train_results  = all_train_results.withColumn("prob_pos", extract_prob_udf(col("probability")))

train_cutoff_df = FP_TP_dataframe(all_train_results, CutOffs)
val_cutoff_df = FP_TP_dataframe(all_val_results, CutOffs)

#plot_precision_recall(train_cutoff_df, val_cutoff_df)

# COMMAND ----------

# DBTITLE 1,Mean encoding for categorical feautres
#  def mean_encode_time_series(df: DataFrame, 
#                         categorical_columns: list, 
#                         target_column: str,
#                         date_column: str):
#     """
#     Performs mean encoding on specified categorical columns in a time-series Spark DataFrame.
#     Uses data up to the current row's date to restrict mean calculation to current date or prior. 
#     """
#     df= df.withColumn('time_long', df.sched_depart_date_time_UTC.cast("long"))\
#                                       .orderBy(df.sched_depart_date_time_UTC)
#     # initialize list to track mean-encoded features
#     mean_enc_features = []

#     # creating hours function
#     hours = lambda i: i * 3600

#     window_spec = Window.partitionBy('TAIL_NUM').orderBy(col('time_long')).rangeBetween(-hours(26), -hours(2))

#     # iterate over categorical features in subset
#     for cat_col in categorical_columns:
#         print("Cat Col: ", cat_col)
#         # define window spec ordered by date:range of 24 hours prior to 2-hour prediction cutoff v
#         windowSpec = Window.partitionBy(cat_col).orderBy(date_column).rangeBetween(-hours(26), -hours(2)) 
#         # calc cumulative mean
#         df = df.withColumn(f"{cat_col}_mean_encoded", mean(target_column).over(windowSpec))
#         # drop original categorical column
#         df = df.drop(cat_col)
#     return df 

# COMMAND ----------

# DBTITLE 1,Grid search
# param_grid = ParamGridBuilder() \
#     .addGrid(pipeline3.maxIter, [100]) \ # .addGrid(getattr(clf,x1), [0.1, 0.2])
#         .addGrid(pipeline3.tol, [1e-06]) \ # .addGrid(getattr(clf,x2),[5,10])\
#             .addGrid(pipeline3.stepSize, [0.01]) \
#                 .addGrid(pipeline3.layers, [layers2]) \
#                     .build()

# crossval = CrossValidator(estimator=pipeline3,
#                       estimatorParamMaps=paramGrid,
#                       evaluator=MulticlassClassificationEvaluator(metricName="precisionByLabel"),
#                       numFolds=3)

# Printing out the parameters you want:
# print 'Best Param (regParam): ', bestModel._java_obj.getstepSize()
# print 'Best Param (MaxIter): ', bestModel._java_obj.getMaxIter()
# print 'Best Param (elasticNetParam): ', bestModel._java_obj.getElasticNetParam()
                                    #  maxIter=100, 
                                    #  tol=1e-06, 
                                    #  seed=1331, 
                                    #  layers=layers1, 
                                    #  blockSize=128, 
                                    #  stepSize=0.03, 
                                    #  solver='l-bfgs', 
                                    
# crossval = CrossValidator(estimator=pipeline,
#                                       estimatorParamMaps=paramGrid,
#                                       evaluator=evaluator,
#                                       numFolds=2)
# cvModel = crossval.fit(train_data)
# cvModel.bestModel.stages[0]._java_obj.getMaxIter()
