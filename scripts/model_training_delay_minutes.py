#!/usr/bin/env python
import os
import io
import joblib
import boto3
from dotenv import load_dotenv
import redshift_connector
import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Constants
S3_BUCKET = "airline-is459"
S3_MODEL_PATH = "data-source/BQ2models/xgb_total_delay_model.joblib"
ENV_FILE_PATH = ".env"

def download_env_file_from_s3(bucket, file_path):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=file_path)
    return obj['Body'].read().decode('utf-8')

# Establish Redshift connection
def connect_to_redshift():
    return redshift_connector.connect(
        host=REDSHIFT_HOST,
        database=REDSHIFT_DB,
        port=REDSHIFT_PORT,
        user=REDSHIFT_USER,
        password=REDSHIFT_PASSWORD
    )

# Execute SQL queries
def execute_sql_query(conn, sql_query):
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        print("SQL query executed successfully.")

# UNLOAD query to S3
def unload_to_s3(conn, table_name, s3_path):
    unload_query = f"""
    UNLOAD ('SELECT * FROM {table_name}')
    TO '{s3_path}'
    IAM_ROLE '{IAM_ROLE_ARN}'
    FORMAT AS PARQUET
    ALLOWOVERWRITE;
    """
    execute_sql_query(conn, unload_query)

# Data preprocessing with Dask
def load_and_prepare_data(s3_path):
    ddf = dd.read_parquet(s3_path, engine='pyarrow', storage_options={'anon': False})
    return ddf.dropna()

# Model training and uploading to S3
def train_and_upload_model(ddf, s3_client, bucket, model_path, existing_booster=None):
    client = Client(n_workers=4)
    X = ddf.drop(columns=['depdelay', 'arrdelay'])
    y = ddf['depdelay'] + ddf['arrdelay']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
    
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    output = xgb.dask.train(client, params, dtrain, num_boost_round=100, xgb_model=existing_booster)
    updated_booster = output['booster']
    
    # Save and upload model
    model_buffer = io.BytesIO()
    joblib.dump(updated_booster, model_buffer)
    model_buffer.seek(0)
    
    s3_client.upload_fileobj(model_buffer, bucket, model_path)
    print(f"Model uploaded to s3://{bucket}/{model_path}")
    
    client.close()
    return updated_booster

# Model evaluation
def evaluate_model(client, booster, X_test, y_test):
    dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)
    y_pred = xgb.dask.predict(client, booster, dtest)
    
    y_test_values = y_test.compute()
    y_pred_values = y_pred.compute()
    mse = mean_squared_error(y_test_values, y_pred_values)
    mae = mean_absolute_error(y_test_values, y_pred_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_values, y_pred_values)
    
    print("\nModel Performance on Test Data:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")

# Main process
def main():
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

    # Debugging: Print environment variables to ensure they are loaded
    print("REDSHIFT_HOST:", REDSHIFT_HOST)
    print("REDSHIFT_DB:", REDSHIFT_DB)
    print("REDSHIFT_PORT:", REDSHIFT_PORT)
    print("REDSHIFT_USER:", REDSHIFT_USER)
    print("REDSHIFT_PASSWORD:", REDSHIFT_PASSWORD)
    print("S3_BUCKET:", S3_BUCKET)
    print("IAM_ROLE_ARN:", IAM_ROLE_ARN)
    
    # Step 2: Connect to Redshift and execute SQL
    conn = connect_to_redshift()
    drop_table_sql = """
    DROP TABLE IF EXISTS bq2_data.historical_feature_engineered_data;
    """
    execute_sql_query(conn, drop_table_sql)

    feature_engineering_sql = """
    CREATE TABLE bq2_data.historical_feature_engineered_data AS
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
            (f.deptime - f.crsdeptime)::FLOAT AS scheduled_dep_diff,
            (f.arrtime - f.crsarrtime)::FLOAT AS scheduled_arr_diff,
            CASE WHEN f.month IN (6, 7, 8, 12) THEN 1 ELSE 0 END AS is_peak_season,
            CASE WHEN f.dayofweek IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,
            (f.origin_temperature_2m - f.dest_temperature_2m)::FLOAT AS temp_diff,
            (f.origin_relative_humidity_2m - f.dest_relative_humidity_2m)::FLOAT AS humidity_diff,
            (f.origin_precipitation - f.dest_precipitation)::FLOAT AS precip_diff,
            f.distance::FLOAT AS distance,
            c.carrier_count::FLOAT / t.total_count AS uniquecarrier_freq,
            f.depdelay,
            f.arrdelay,
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
            f.dest_wind_gusts_10m
        FROM BQ2_data.joined_flights_full_weather f
        JOIN carrier_counts c ON f.uniquecarrier = c.uniquecarrier
        CROSS JOIN total_count t
    )
    SELECT
        ROW_NUMBER() OVER () AS id,
        *
    FROM feature_engineered;
    """
    execute_sql_query(conn, feature_engineering_sql)
    unload_to_s3(conn, "bq2_data.historical_feature_engineered_data", f"s3://{S3_BUCKET}/data-source/BQ2/historical_feature_engineered_data/")
    conn.close()
    
    # Step 3: Load data from S3 and preprocess
    ddf = load_and_prepare_data(f"s3://{S3_BUCKET}/data-source/BQ2/historical_feature_engineered_data/")
    
    # Step 4: Check if model exists in S3
    model_buffer = io.BytesIO()
    existing_booster = None
    try:
        s3_client.download_fileobj(S3_BUCKET, S3_MODEL_PATH, model_buffer)
        model_buffer.seek(0)
        existing_booster = joblib.load(model_buffer)
        print("Existing model loaded from S3.")
    except Exception as e:
        print(f"Failed to load model from S3: {e}. Proceeding with new training.")
    
    # Step 5: Train model and upload
    booster = train_and_upload_model(ddf, s3_client, S3_BUCKET, S3_MODEL_PATH, existing_booster)
    
    # Step 6: Evaluate model
    client = Client(n_workers=4)
    X = ddf.drop(columns=['depdelay', 'arrdelay'])
    y = ddf['depdelay'] + ddf['arrdelay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    evaluate_model(client, booster, X_test, y_test)
    client.close()

if __name__ == "__main__":
    main()
