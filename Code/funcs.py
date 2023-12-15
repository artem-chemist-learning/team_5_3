import pandas as pd
import numpy as np
import seaborn as sns
from itertools import combinations
from databricks.sdk.runtime import *
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("W261").getOrCreate()
from pyspark.sql import functions as f
from pyspark.sql import Window as W

def describe_table(df):
    '''
    This function takes a spark dataframe as an argument and 
    returns a transposed dataframe of the features and their describe metrics.
    Results are provided in descending order of value counts.
    NOTE: This is not an efficient function and can take some time to run.
    Results are not cached within function either. 
    '''
    # creating table of feature describe()
    feature_sample = df.describe()

    # converting feature sample to pandas df and transposing
    feature_sample = feature_sample.toPandas().T

    # promoting first row to headers, and dropping row
    feature_sample.columns = feature_sample.iloc[0]
    feature_sample = feature_sample.drop(feature_sample.index[0])

    # casting count to numeric & sorting in descending order
    feature_sample['count'] = pd.to_numeric(feature_sample['count'])
    feature_sample = feature_sample.sort_values(by='count', ascending=False)

    #resetting index
    feature_sample = feature_sample.reset_index()
    return feature_sample

def blob_connect():
    '''
    This function connects to our team blob storage.
    '''
    blob_container  = "team53container"       # The name of your container created in https://portal.azure.com
    storage_account = "w261team53"  # The name of your Storage account created in https://portal.azure.com
    secret_scope    = "team53scope"           # The name of the scope created in your local computer using the Databricks CLI
    secret_key      = "team53secret"             # The name of the secret key created in your local computer using the Databricks CLI
    team_blob_url   = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket

    # SAS Token: Grant the team limited access to Azure Storage resources
    spark.conf.set(
    f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
    dbutils.secrets.get(scope = secret_scope, key = secret_key)
    )
    return team_blob_url

def write_parquet_to_blob(df, location):
    '''
    This function writes a dataframe to our team's blob storage
    at the location passed in as an argument.
    '''
    # connect to blob
    team_blob_url = blob_connect()

    # write to blob
    df.write.mode('overwrite').parquet(f"{team_blob_url}/{location}")

def split_by_report_type(df):
    '''
    This function takes in a dataset as an argument and splits it into the following datasets:
        df_SOD: This dataframe consists of all daily summaries. (report_type = 'SOD')
        df_SOM: This dataframe consists of all monthly summaries. (report_type = 'MOD')
        df_observations: This dataframe includes all records of report types other 
            than the two mentioned above. 
    '''

    # df of just the MOD values
    df_SOM = df.filter(df.REPORT_TYPE.contains('SOM'))

    # df of just the SOD values
    df_SOD = df.filter(df.REPORT_TYPE.contains('SOD'))

    # Remove leading and trailing whitespaces in 'REPORT_TYPE'
    df = df.withColumn('REPORT_TYPE', f.trim(df.REPORT_TYPE))
    
    # df of remaining values
    df = df.filter(~df['REPORT_TYPE'].isin(['SOD', 'SOM']))

    return df_SOD, df_SOM, df

def drop_empty_cols(df):
    '''
    This function takes in the dataframe and drops
    all columns with no values. 
    '''
    # creating a pandas df of the summary stats for SOD df
    df_sum =  describe_table(df)

    # identifying the null columns
    drop_cols = df_sum['index'][df_sum['count'] == 0].to_list()

    # dropping null columns from df
    df = df.drop(*drop_cols)
    return df

def pairplot(df, features, sample_size=3000):
    '''
    Calculates the Pearson Correlation Coefficient for every possible combination of features specified.
    Takes a defined sample size of the dataframe and creates a pairplot of the features specified. 
    '''
    print("="*60)
    print("Pearson Correlation Coefficients")
    print('-'*60)
    for combo in combinations(features, 2):
        print(combo, end=":  ")
        pearcorr = df.corr(combo[0], combo[1])
        print(pearcorr)
    print('='*60)


    # quantifying total records
    total_records = df.count()

    # determining sampling fraction
    if sample_size > total_records:
        fraction = 1.0
    else:
        fraction = sample_size/total_records
    print("Pairplot of Sample")
    print('-'*60)
    print("Total records: ", total_records)
    print("Records plotted: ", sample_size)
    print("Fraction used for Plot: ", fraction)
    
    
    # sampling df for pandas visualizations
    data_sample = df.sample(fraction = fraction).toPandas()

    # features for plot
    features_to_plot = data_sample[features]

    # Basic correlogram
    sns.pairplot(features_to_plot,
                vars = features,
                corner = True,
                diag_kws = {'bins':20}) 
    
def histogram(df, feature, sample_size=2000, bins = 20):
    '''
    Creates a histogram for a sampling of the dataframe and feature specified.
    '''

    # quantifying total records
    total_records = df.count()

    # determining sampling fraction
    if sample_size > total_records:
        fraction = 1.0
    else:
        fraction = sample_size/total_records
    
    # sampling df for pandas visualizations
    data_sample = df.sample(fraction = fraction).toPandas()

    # feature to plot
    feature_to_plot = data_sample[feature]

    # Basic correlogram
    sns.histplot(feature_to_plot,
                bins = bins)
    
def drop_existing_weather(df):
    '''Function for dropping the pre-existing weather features from OTPW dataset.'''
    drop_columns = ['STATION',
                'DATE',
                'LATITUDE',
                'LONGITUDE',
                'ELEVATION',
                'NAME',
                'REPORT_TYPE',
                'SOURCE',
                'HourlyAltimeterSetting',
                'HourlyDewPointTemperature',
                'HourlyDryBulbTemperature',
                'HourlyPrecipitation',
                'HourlyPresentWeatherType',
                'HourlyPressureChange',
                'HourlyPressureTendency',
                'HourlyRelativeHumidity',
                'HourlySkyConditions',
                'HourlySeaLevelPressure',
                'HourlyStationPressure',
                'HourlyVisibility',
                'HourlyWetBulbTemperature',
                'HourlyWindDirection',
                'HourlyWindGustSpeed',
                'HourlyWindSpeed',
                'Sunrise',
                'Sunset',
                'DailyAverageDewPointTemperature',
                'DailyAverageDryBulbTemperature',
                'DailyAverageRelativeHumidity',
                'DailyAverageSeaLevelPressure',
                'DailyAverageStationPressure',
                'DailyAverageWetBulbTemperature',
                'DailyAverageWindSpeed',
                'DailyCoolingDegreeDays',
                'DailyDepartureFromNormalAverageTemperature',
                'DailyHeatingDegreeDays',
                'DailyMaximumDryBulbTemperature',
                'DailyMinimumDryBulbTemperature',
                'DailyPeakWindDirection',
                'DailyPeakWindSpeed',
                'DailyPrecipitation',
                'DailySnowDepth',
                'DailySnowfall',
                'DailySustainedWindDirection',
                'DailySustainedWindSpeed',
                'DailyWeather',
                'MonthlyAverageRH',
                'MonthlyDaysWithGT001Precip',
                'MonthlyDaysWithGT010Precip',
                'MonthlyDaysWithGT32Temp',
                'MonthlyDaysWithGT90Temp',
                'MonthlyDaysWithLT0Temp',
                'MonthlyDaysWithLT32Temp',
                'MonthlyDepartureFromNormalAverageTemperature',
                'MonthlyDepartureFromNormalCoolingDegreeDays',
                'MonthlyDepartureFromNormalHeatingDegreeDays',
                'MonthlyDepartureFromNormalMaximumTemperature',
                'MonthlyDepartureFromNormalMinimumTemperature',
                'MonthlyDepartureFromNormalPrecipitation',
                'MonthlyDewpointTemperature',
                'MonthlyGreatestPrecip',
                'MonthlyGreatestPrecipDate',
                'MonthlyGreatestSnowDepth',
                'MonthlyGreatestSnowDepthDate',
                'MonthlyGreatestSnowfall',
                'MonthlyGreatestSnowfallDate',
                'MonthlyMaxSeaLevelPressureValue',
                'MonthlyMaxSeaLevelPressureValueDate',
                'MonthlyMaxSeaLevelPressureValueTime',
                'MonthlyMaximumTemperature',
                'MonthlyMeanTemperature',
                'MonthlyMinSeaLevelPressureValue',
                'MonthlyMinSeaLevelPressureValueDate',
                'MonthlyMinSeaLevelPressureValueTime',
                'MonthlyMinimumTemperature',
                'MonthlySeaLevelPressure',
                'MonthlyStationPressure',
                'MonthlyTotalLiquidPrecipitation',
                'MonthlyTotalSnowfall',
                'MonthlyWetBulb',
                'AWND',
                'CDSD',
                'CLDD',
                'DSNW',
                'HDSD',
                'HTDD',
                'NormalsCoolingDegreeDay',
                'NormalsHeatingDegreeDay',
                'ShortDurationEndDate005',
                'ShortDurationEndDate010',
                'ShortDurationEndDate015',
                'ShortDurationEndDate020',
                'ShortDurationEndDate030',
                'ShortDurationEndDate045',
                'ShortDurationEndDate060',
                'ShortDurationEndDate080',
                'ShortDurationEndDate100',
                'ShortDurationEndDate120',
                'ShortDurationEndDate150',
                'ShortDurationEndDate180',
                'ShortDurationPrecipitationValue005',
                'ShortDurationPrecipitationValue010',
                'ShortDurationPrecipitationValue015',
                'ShortDurationPrecipitationValue020',
                'ShortDurationPrecipitationValue030',
                'ShortDurationPrecipitationValue045',
                'ShortDurationPrecipitationValue060',
                'ShortDurationPrecipitationValue080',
                'ShortDurationPrecipitationValue100',
                'ShortDurationPrecipitationValue120',
                'ShortDurationPrecipitationValue150',
                'ShortDurationPrecipitationValue180',
                'REM',
                'BackupDirection',
                'BackupDistance',
                'BackupDistanceUnit',
                'BackupElements',
                'BackupElevation',
                'BackupEquipment',
                'BackupLatitude',
                'BackupLongitude',
                'BackupName',
                'WindEquipmentChangeDate',]
    reduced_df = df.drop(*drop_columns)
    return reduced_df

def add_prefix(df, prefix):
    '''Function for adding prefixes to every column in a dataset.'''
    for column in df.columns:
        df = df.withColumnRenamed(column, '{}{}'.format(prefix, column))
    return df

def joining_hourly(df, weather_df, prefix):
    '''Function for joining the hourly weather data to flight data, based on prefix.'''
    # adding prefixes to values
    prefixed_weather = add_prefix(weather_df, f"{prefix}_")

    # registering dataframes
    df.createOrReplaceTempView("df")
    prefixed_weather.createOrReplaceTempView("prefixed_weather")
    
    joined_df = df.join(
            prefixed_weather,
            on=[
                df[f'{prefix}_station_id'] == prefixed_weather[f'{prefix}_STATION'],
                df['four_hours_prior_depart_UTC'] <= prefixed_weather[f'{prefix}_UTC'],
                df['three_hours_prior_depart_UTC'] > prefixed_weather[f'{prefix}_UTC'] 
            ],
            how='left'
    # rank weather UTC's 
    # based on the dates that already filtered by event's date
    ).withColumn('rank_UTC', f.rank().over(W.partitionBy(f'{prefix}_station_id', 'three_hours_prior_depart_UTC').orderBy(prefixed_weather[f'{prefix}_UTC'].desc()))) \
    .where(f.col('rank_UTC') == 1) \
    .drop('rank_UTC')
    return joined_df

# -------------END ERIK'S FUNCTIONS -----------------------------------------------------------------------  



# -------------START LUCY EDA/CLEANING FUNCTIONS: ----------------------------------------------------------

#bailey edit so this returns something instead of prints
def get_df_dimensions(spark_df):
    """Returns num columns, num rows"""
    #print(f"Number of columns in dataframe: {len(spark_df.columns)}")
    #print(f"Number of rows in dataframe: {spark_df.count()}")
    return len(spark_df.columns), spark_df.count()

def cast_cols_to_new_type(spark_df, columns_to_cast):
    '''Cast columns in dataframe to new types based on comprehensive columns dictionary.'''
    # define the expressions for each column
    cast_exprs = [f"cast({col} as {new_type}) as {col}" for col, new_type in columns_to_cast.items()]
    # apply the cast expressions
    spark_df = spark_df.selectExpr(*cast_exprs)
    #spark_df.printSchema()
    return spark_df

def count_missing_values(spark_df, sort=True):
    """
    Calculates the percentage of null/nan values in each column of a Spark dataframe.
    """
    # calculate total number of rows in df 
    total_rows = spark_df.count()
    # list comprehension to create a list of formatted column expressions
    missing_value_counts = [
        (F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)) / total_rows * 100).alias(c)
        for c, c_type in spark_df.dtypes if c_type not in ('timestamp', 'date')
    ]
    # select expressions in the dataframe
    missing_df = spark_df.select(missing_value_counts).toPandas()
    if missing_df.empty:
        print("No missing values in this dataframe.")
        return None
    if sort:
        missing_df = missing_df.rename(index={0: 'null_percent'}).T.sort_values("null_percent", ascending=False)
    return missing_df

def drop_cols_by_null_prct(missing_df, main_df):
    # missing_df.reset_index(inplace=True)
    # missing_df.rename(columns={'index': 'column_name'}, inplace=True)
    columns_to_drop = missing_df[missing_df['null_percent'] >= 90]['column_name']
    columns_to_drop = columns_to_drop.tolist()
    #drop from main spark df
    main_df = main_df.drop(*columns_to_drop)
    return main_df

def filter_df_by_min_max(df, column, min_val, max_val=None):
    """
    Filters a Pandas DataFrame based on a minimum and an optional maximum value in a specified column.
    :param df: Pandas DataFrame to filter
    :param column: Column name to apply the filter on
    :param min_val: Minimum value for filtering
    :param max_val: Optional maximum value for filtering, defaults to None
    :return: Filtered DataFrame
    """
    if max_val is not None:
        filtered_df = df[(df[column] >= min_val) & (df[column] <= max_val)]
    else:
        filtered_df = df[df[column] >= min_val]
    return filtered_df


def filter_multiple_associations(df, main_col: str, assoc_col: str):
    """
    Filter rows where values in the main_col have more than one distinct value associated in assoc_col.
    :param df: The Spark DataFrame to operate on
    :param main_col: The column to analyze
    :param assoc_col: The column to check for multiple associations
    :return: A DataFrame with rows where the main_col values have more than one distinct assoc_col value
    """
    # froup by main column and count distinct values in associated column
    count_df = df.groupBy(main_col).agg(F.countDistinct(assoc_col).alias('distinct_count'))
    # filter rows where count of distinct values is greater than 1
    filtered_df = count_df.filter(F.col('distinct_count') > 1).select(main_col)
    return filtered_df

def plot_boxplots(df, columns, sample_fraction=0.1):
    """
    Plots individual boxplots for multiple Spark DataFrame columns.
    :param df: The Spark DataFrame
    :param columns: A list of column names to plot
    :param sample_fraction: Fraction of data to sample for plotting (default 0.1)
    """
    # Sample the data if the DataFrame is large
    sampled_df = df.select(columns).sample(False, sample_fraction).toPandas()
    # Number of columns (for subplots arrangement)
    num_cols = len(columns)
    # Create a figure and axes
    fig, axes = plt.subplots(num_cols, 1, figsize=(8, 4 * num_cols))
    # Plot each column
    for i, col in enumerate(columns):
        sns.boxplot(x=sampled_df[col], ax=axes[i] if num_cols > 1 else axes)
        axes[i].set_title(f'Boxplot of {col}' if num_cols > 1 else 'Boxplot')
    plt.tight_layout()
    plt.show()
    
def calculate_percentage_difference(df, col1: str, col2: str) -> float:
    """
    Calculates the percentage of rows in a DataFrame where two columns have different values.

    :param df: The Spark DataFrame to analyze
    :param col1: The name of the first column
    :param col2: The name of the second column
    :return: The percentage of rows with different values in the two columns
    """
    # Count the total number of rows
    total_rows = df.count()
    # Count the number of rows where the values in the two columns are different
    diff_count = df.filter(F.col(col1) != F.col(col2)).count()
    # Calculate the percentage
    percentage_diff = (diff_count / total_rows) * 100
    return percentage_diff

def compare_specific_delay_values(main_df,specific_delay_col):
    total_rows=main_df.count()
    specific_delay_df = main_df.select('DEP_DELAY','DEP_DELAY_NEW',specific_delay_col)\
        .filter(F.col(specific_delay_col).isNotNull())
    print(f"Percentage of non-null {specific_delay_col} observations: {(specific_delay_df.count()/total_rows)*100}")
    specific_delay_df_non0 = specific_delay_df.filter(F.col('CARRIER_DELAY')!=0)
    print(f"Percentage of non-0 {specific_delay_col} observations: {(specific_delay_df_non0.count()/total_rows)*100}")
    # count number of non-null (incl. 0) rows where values in two columns are different
    specific_delay_df = specific_delay_df.filter(F.col('DEP_DELAY')==F.col(specific_delay_col)).count()
    # calculate percentage
    print(f"Percentage of {specific_delay_col} values matching 'DEP_DELAY':{(specific_delay_df_non0.count()/total_rows)*100}")

def make_lookup_table(main_df, lookup_col_list):
    temp_df = main_df.select(lookup_col_list)
    distinct_df = temp_df.distinct().toPandas()
    return distinct_df

def write_pd_df_to_storage(pd_df,location):
    spark_df = spark.createDataFrame(pd_df)
    spark_df.write.mode('overwrite').parquet(f"{team_blob_url}/{location}")

def calculate_time_difference(end_time_col, start_time_col):
    """
    Calculates the time difference in minutes for columns in numeric HHMM or HMM format, accounting for day changes.
    """
    # extract hours and minutes from the float format
    end_time_col_hours = F.floor(end_time_col / 100)
    end_time_col_minutes = end_time_col % 100
    start_time_col_hours = F.floor(start_time_col / 100)
    start_time_col_minutes = start_time_col % 100
    # convert to total minutes
    end_time_col_total = (end_time_col_hours * 60) + end_time_col_minutes
    start_time_col_total = (start_time_col_hours * 60) + start_time_col_minutes
    # check for day change (end time up to 2am but start time after 8pm)
    condition = (end_time_col_hours < 3) & (start_time_col_hours > 20)
    # add 24 hours (1440 minutes) to end_time_col_total if day change
    return F.when(condition, end_time_col_total + 1440).otherwise(end_time_col_total) - start_time_col_total

def make_feature_subset_pearson_corr_heatmap(spark_df, str_to_select_cols,plot_title):
    # list all columns in spark df
    all_columns = spark_df.columns
    # select columns containing string that identifies the feature subset
    heatmap_columns = [col for col in all_columns if str_to_select_cols in col]
    # subset those columns in spark df
    df_filtered = spark_df.select(heatmap_columns)
    # convert to pandas df
    pandas_df = df_filtered.toPandas()
    # calculate the correlation matrix
    corr_matrix = pandas_df.corr()
    # set repeated values in upper triangle of correlation matrix to NaN (simplify plot)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix[mask] = np.nan
    # plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(plot_title)
    plt.show()

# *more efficient heatmap function -------
#from pyspark.sql.functions import col, corr
def make_feature_subset_pearson_corr_heatmap_eff(spark_df, str_to_select_cols, plot_title):
    # Select columns containing string that identifies the feature subset
    heatmap_columns = [col for col in spark_df.columns if str_to_select_cols in col]
    # Calculate the correlation matrix directly in Spark
    corr_matrix = None
    for i in range(len(heatmap_columns)):
        for j in range(i + 1, len(heatmap_columns)):
            col_i = heatmap_columns[i]
            col_j = heatmap_columns[j]
            corr_value = spark_df.select(F.corr(F.col(col_i), F.col(col_j)).alias(f"{col_i}_{col_j}"))
            # Aggregate the results into a correlation matrix (you'll need to implement this part)

    # Convert the correlation matrix to Pandas for plotting (if the matrix is small enough)
    pandas_corr_matrix = corr_matrix.toPandas()

    # Plot the heatmap (same as your original code)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pandas_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(plot_title)
    plt.show()

def check_column_consistency(df, col1: str, col2: str, col3: str = None):
    """
    Checks if the values in three specified columns of a Spark DataFrame are consistent.

    Parameters:
    df (DataFrame): The Spark DataFrame to check.
    col1 (str): The name of the first column to compare.
    col2 (str): The name of the second column to compare.
    col3 (str): (OPTIONAL) The name of the third column to compare.

    Returns:
    DataFrame: A DataFrame containing the inconsistencies, if any.
    """
    # find unique combinations of the specified columns
    unique_combinations = df.select(col1, col2, col3).distinct()
    # count each combination
    combination_counts = unique_combinations.groupBy(col1, col2, col3).count()
    # filter for combinations where count > 1
    inconsistencies = combination_counts.filter(F.col("count") > 1)
    return inconsistencies

def print_all_distinct_values(df, column_name):
    distinct_values = df.select(column_name).distinct().collect()
    print(distinct_values)


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
    window_spec = W.partitionBy().orderBy(split_feature)

    # creating a rank column for ordered df
    ranked_df = df.withColumn("rank", f.percent_rank().over(window_spec))
    
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
