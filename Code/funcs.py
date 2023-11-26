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