#!/usr/bin/env python
import os
import redshift_connector
import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import boto3
from dotenv import load_dotenv
import io

# Constants
S3_BUCKET = 'airline-is459'
S3_MODEL_PATH = "data-source/BQ2models/xgb_multi_label_model.joblib"
ENV_FILE_PATH = ".env"

def download_env_file_from_s3(bucket, file_path):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=file_path)
    return obj['Body'].read().decode('utf-8')

# Execute SQL queries
def execute_sql_query(conn, sql_query):
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        print("SQL query executed successfully.")

# Establish Redshift connection
def connect_to_redshift():
    return redshift_connector.connect(
        host=REDSHIFT_HOST,
        database=REDSHIFT_DB,
        port=REDSHIFT_PORT,
        user=REDSHIFT_USER,
        password=REDSHIFT_PASSWORD
    )

# Step 1: Connect to Redshift and create a feature-engineered table
def create_feature_engineered_table():
    conn = connect_to_redshift()
    drop_table_sql = """
    DROP TABLE IF EXISTS bq2_data.historical_predict_delay_type;
    """
    execute_sql_query(conn, drop_table_sql)
    
    sql_create_table = """
    CREATE TABLE bq2_data.historical_predict_delay_type AS
    WITH carrier_counts AS (
        SELECT uniquecarrier, COUNT(*) AS carrier_count
        FROM bq2_data.joined_flights_full_weather
        GROUP BY uniquecarrier
    ),
    total_count AS (
        SELECT COUNT(*) AS total_count FROM bq2_data.joined_flights_full_weather
    ),
    feature_engineered AS (
        SELECT
            FLOOR(f.crsdeptime / 100)::INT AS departure_hour,
            FLOOR(f.crsarrtime / 100)::INT AS arrival_hour,
            CASE WHEN f.month IN (6, 7, 8, 12) THEN 1 ELSE 0 END AS is_peak_season,
            CASE WHEN f.dayofweek IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,
            (f.origin_temperature_2m - f.dest_temperature_2m)::FLOAT AS temp_diff,
            (f.origin_relative_humidity_2m - f.dest_relative_humidity_2m)::FLOAT AS humidity_diff,
            (f.origin_precipitation - f.dest_precipitation)::FLOAT AS precip_diff,
            f.origin_temperature_2m,
            f.origin_relative_humidity_2m,
            f.origin_dew_point_2m,
            f.origin_precipitation,
            f.origin_snow_depth,
            f.origin_pressure_msl,
            f.origin_surface_pressure,
            f.origin_cloud_cover,
            f.origin_wind_speed_10m,
            f.origin_wind_direction_10m,
            f.origin_wind_gusts_10m,
            f.dest_temperature_2m,
            f.dest_relative_humidity_2m,
            f.dest_dew_point_2m,
            f.dest_precipitation,
            f.dest_snow_depth,
            f.dest_pressure_msl,
            f.dest_surface_pressure,
            f.dest_cloud_cover,
            f.dest_wind_speed_10m,
            f.dest_wind_direction_10m,
            f.dest_wind_gusts_10m,
            f.distance::FLOAT AS distance,
            c.carrier_count::FLOAT / t.total_count AS uniquecarrier_freq,
            CASE WHEN f.depdelay > 0 THEN 1 ELSE 0 END AS depdelay_occurred,
            CASE WHEN f.arrdelay > 0 THEN 1 ELSE 0 END AS arrdelay_occurred,
            CASE WHEN f.weatherdelay > 0 THEN 1 ELSE 0 END AS weatherdelay_occurred,
            CASE WHEN f.nasdelay > 0 THEN 1 ELSE 0 END AS nasdelay_occurred,
            CASE WHEN f.securitydelay > 0 THEN 1 ELSE 0 END AS securitydelay_occurred,
            CASE WHEN f.lateaircraftdelay > 0 THEN 1 ELSE 0 END AS lateaircraftdelay_occurred,
            CASE WHEN f.carrierdelay > 0 THEN 1 ELSE 0 END AS carrierdelay_occurred
        FROM bq2_data.joined_flights_full_weather f
        JOIN carrier_counts c ON f.uniquecarrier = c.uniquecarrier
        CROSS JOIN total_count t
    )
    SELECT
        ROW_NUMBER() OVER () AS id,
        *
    FROM feature_engineered;
    """
    execute_sql_query(conn, sql_create_table)
    conn.close()

# Step 2: Unload data from Redshift to S3
def unload_data_to_s3():
    conn = connect_to_redshift()
    
    unload_query = """
    UNLOAD ('SELECT * FROM bq2_data.historical_predict_delay_type')
    TO 's3://airline-is459/data-source/BQ2/classification_historical_features_engineered_data/'
    IAM_ROLE '{}'
    FORMAT AS PARQUET
    ALLOWOVERWRITE;
    """.format(IAM_ROLE_ARN)
    
    execute_sql_query(conn, unload_query)
    conn.close()

# Step 3: Load data from S3 and preprocess with Dask
def load_and_prepare_data(s3_path):
    local_dir = "/tmp/data"
    os.makedirs(local_dir, exist_ok=True)
    s3_client = boto3.client("s3")
    bucket = s3_path.split('/')[2]
    prefix = '/'.join(s3_path.split('/')[3:])
    
    # List and download files
    objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in objects.get('Contents', []):
        file_key = obj['Key']
        file_name = os.path.join(local_dir, file_key.split('/')[-1])
        s3_client.download_file(bucket, file_key, file_name)
    
    # Load data from local files
    ddf = dd.read_parquet(local_dir, engine='pyarrow')
    return ddf.dropna()

# Step 4: Define feature and target columns
def define_features_targets(ddf):
    feature_columns = ['departure_hour', 'arrival_hour', 'is_peak_season', 'is_weekend', 'temp_diff', 'humidity_diff', 'precip_diff', 'distance', 'uniquecarrier_freq',
                       'origin_temperature_2m', 'origin_relative_humidity_2m', 'origin_dew_point_2m', 'origin_precipitation', 'origin_snow_depth', 'origin_pressure_msl',
                       'origin_surface_pressure', 'origin_cloud_cover', 'origin_wind_speed_10m', 'origin_wind_direction_10m', 'origin_wind_gusts_10m',
                       'dest_temperature_2m', 'dest_relative_humidity_2m', 'dest_dew_point_2m', 'dest_precipitation', 'dest_snow_depth', 'dest_pressure_msl',
                       'dest_surface_pressure', 'dest_cloud_cover', 'dest_wind_speed_10m', 'dest_wind_direction_10m', 'dest_wind_gusts_10m']
    target_columns = ['depdelay_occurred', 'arrdelay_occurred', 'weatherdelay_occurred', 'nasdelay_occurred', 'securitydelay_occurred', 'lateaircraftdelay_occurred', 'carrierdelay_occurred']
    
    X = ddf[feature_columns]
    y = ddf[target_columns]
    return X, y

# Step 5: Split data into training and testing sets
def split_data(X, y):
    client = Client(n_workers=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test, client

# Step 6: Train the multi-label classification model
def train_model(X_train, y_train):
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=-1, verbosity=1)
    model = MultiOutputClassifier(xgb_clf, n_jobs=-1)
    
    X_train_pandas = X_train.compute()
    y_train_pandas = y_train.compute()
    
    model.fit(X_train_pandas, y_train_pandas)
    return model

# Step 7: Evaluate the model
def evaluate_model(model, X_test, y_test, target_columns):
    X_test_pandas = X_test.compute()
    y_test_pandas = y_test.compute()
    
    y_pred = model.predict(X_test_pandas)
    print("\nClassification Report:")
    print(classification_report(y_test_pandas, y_pred, target_names=target_columns))

# Step 8: Save the trained model to S3
def save_model_to_s3(model):
    s3_client = boto3.client('s3')
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)
    
    try:
        s3_client.upload_fileobj(model_buffer, S3_BUCKET, S3_MODEL_PATH)
        print(f"Model uploaded to s3://{S3_BUCKET}/{S3_MODEL_PATH}")
    except Exception as e:
        print(f"Failed to upload model to S3: {e}")

# Step 9: Main execution sequence
if __name__ == "__main__":
    s3_client = boto3.client("s3")
    
    # Step 1: Download and load the .env file from S3
    env_content = download_env_file_from_s3(S3_BUCKET, ENV_FILE_PATH)
    with open(ENV_FILE_PATH, 'w') as f:
        f.write(env_content)
    load_dotenv(ENV_FILE_PATH)

    # Constants
    global REDSHIFT_HOST, REDSHIFT_DB, REDSHIFT_PORT, REDSHIFT_USER, REDSHIFT_PASSWORD, IAM_ROLE_ARN
    REDSHIFT_HOST = os.getenv("REDSHIFT_HOST")
    REDSHIFT_DB = os.getenv("REDSHIFT_DB")
    REDSHIFT_PORT = int(os.getenv("REDSHIFT_PORT", 5439))
    REDSHIFT_USER = os.getenv("REDSHIFT_USER")
    REDSHIFT_PASSWORD = os.getenv("REDSHIFT_PASSWORD")
    IAM_ROLE_ARN = os.getenv("IAM_ROLE_ARN")

    # Step 1: Create feature-engineered table
    create_feature_engineered_table()
    
    # Step 2: Unload data to S3
    unload_data_to_s3()
    
    # Step 3: Load data from S3 and preprocess
    ddf = load_and_prepare_data(f"s3://{S3_BUCKET}/data-source/BQ2/classification_historical_features_engineered_data/")
    
    # Step 4: Define features and targets
    X, y = define_features_targets(ddf)
    
    # Step 5: Split data into training and testing sets
    X_train, X_test, y_train, y_test, client = split_data(X, y)
    
    # Step 6: Train the model
    model = train_model(X_train, y_train)
    
    # Step 7: Evaluate the model
    evaluate_model(model, X_test, y_test, target_columns=['depdelay_occurred', 'arrdelay_occurred', 'weatherdelay_occurred', 'nasdelay_occurred', 'securitydelay_occurred', 'lateaircraftdelay_occurred', 'carrierdelay_occurred'])
    
    # Step 8: Save the model to S3
    save_model_to_s3(model)
    
    # Step 9: Shut down Dask client
    client.close()
