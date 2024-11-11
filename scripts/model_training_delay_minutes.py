#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install dask_ml dask[dataframe] dask[distributed] pandas redshift_connector xgboost scikit-learn s3fs joblib


# In[19]:


import redshift_connector
# Establish the connection
conn = redshift_connector.connect(
    host='default-workgroup.820242926303.us-east-1.redshift-serverless.amazonaws.com',
    database='dev',
    port=5439,
    user='bq2user',
    password='Sunrise@8785'  # Replace with your actual password or use a secure method
)

# Enable autocommit
conn.autocommit = True

# Create a cursor object
cursor = conn.cursor()

# SQL query to perform feature engineering
cursor.execute("DROP TABLE IF EXISTS bq2_data.historical_feature_engineered_data;")
sql_query = """
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
    *
FROM feature_engineered;
"""
cursor.execute(sql_query)
print("SQL query executed successfully.")
cursor.close()
conn.close()


# In[20]:


import redshift_connector
# Establish the connection
conn = redshift_connector.connect(
    host='default-workgroup.820242926303.us-east-1.redshift-serverless.amazonaws.com',
    database='dev',
    port=5439,
    user='bq2user',
    password='Sunrise@8785'  # Replace with your actual password or use a secure method
)

# Enable autocommit
conn.autocommit = True

# Create a cursor object
cursor = conn.cursor()

unload_query = """
UNLOAD ('SELECT * FROM bq2_data.historical_feature_engineered_data')
TO 's3://airline-is459/data-source/BQ2/historical_feature_engineered_data/'
IAM_ROLE 'arn:aws:iam::820242926303:role/service-role/AmazonRedshift-CommandsAccessRole-20241017T010122'
FORMAT AS PARQUET
ALLOWOVERWRITE;
"""
# Execute the UNLOAD command
cursor.execute(unload_query)

# Close the cursor and connection
cursor.close()
conn.close()


# In[2]:


import dask.dataframe as dd

# Read the Parquet files from S3
ddf = dd.read_parquet('s3://airline-is459/data-source/BQ2/historical_feature_engineered_data/', storage_options={'anon': False})

# Proceed with data preprocessing and model training
ddf = ddf.dropna()


# In[4]:


from dask.distributed import Client
from dask_ml.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import boto3
import io
import joblib
import pandas as pd


# Initialize Dask client
client = Client(n_workers=4)

# Define features and target
X = ddf.drop(columns=['depdelay', 'arrdelay'])

# Add log transformations for selected features
X['distance_x_departure_hour'] = X['distance'] * X['departure_hour']
X['uniquecarrier_freq_x_distance'] = X['uniquecarrier_freq'] * X['distance']
X['log_distance'] = np.log1p(X['distance'])
X['log_uniquecarrier_freq'] = np.log1p(X['uniquecarrier_freq'])

# Categorical Weather Indicator: Severe Weather
X['severe_weather'] = ((X['origin_precipitation'] > 0.5) | (X['origin_wind_speed_10m'] > 20)).astype(int)

print(X.columns)

# Define target
y = ddf['depdelay'] + ddf['arrdelay']

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
)

# Convert training and validation data to Dask DMatrix
dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
dvalid = xgb.dask.DaskDMatrix(client, X_valid, y_valid)

# Define tuned model parameters without n_estimators, as it's controlled by num_boost_round
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.85,
    'colsample_bytree': 0.8
}

# Train the XGBoost model with early stopping
output = xgb.dask.train(
    client,
    params,
    dtrain,
    num_boost_round=400,  # Increased boosting rounds
    evals=[(dvalid, 'validation')],  # Validation set for early stopping
    early_stopping_rounds=50  # Stops if validation score doesn't improve for 50 rounds
)
booster = output['booster']

# Feature importance analysis
importance = booster.get_score(importance_type='weight')
importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

model_buffer = io.BytesIO()
joblib.dump(booster, model_buffer)
model_buffer.seek(0)

# Set up the S3 client and specify bucket and path
s3_client = boto3.client('s3')
s3_bucket_name = 'airline-is459'  # Replace with your S3 bucket name
s3_model_path = 'data-source/BQ2models/xgb_total_delay_model.joblib'  # Path in the bucket

# Upload buffer directly to S3
try:
    s3_client.upload_fileobj(model_buffer, s3_bucket_name, s3_model_path)
    print(f"Model uploaded to s3://{s3_bucket_name}/{s3_model_path}")
except Exception as e:
    print(f"Failed to upload model to S3: {e}")

# ---- Evaluation on the Test Set ----
# Prepare test data
dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)

# Perform predictions
y_pred = xgb.dask.predict(client, booster, dtest)

# Compute metrics
y_test_values = y_test.compute().copy()
y_pred_values = y_pred.compute().copy()
mse = mean_squared_error(y_test_values, y_pred_values)
mae = mean_absolute_error(y_test_values, y_pred_values)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_values, y_pred_values)

print("\nModel Performance on Test Data:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Display feature importance
print("\nFeature Importance:")
print(importance_df)

client.close()


# BELOW IS TRAINING USING LATEST DATA LOADING MODEL FROM S3 THEN SAVE BACK TO S3

# BELOW IS TRAINING USING LATEST DATA LOADING MODEL FROM S3 THEN SAVE BACK TO S3

# BELOW IS TRAINING USING LATEST DATA LOADING MODEL FROM S3 THEN SAVE BACK TO S3

# BELOW IS TRAINING USING LATEST DATA LOADING MODEL FROM S3 THEN SAVE BACK TO S3

# BELOW IS TRAINING USING LATEST DATA LOADING MODEL FROM S3 THEN SAVE BACK TO S3

# In[8]:


import redshift_connector
# Establish the connection
conn = redshift_connector.connect(
    host='default-workgroup.820242926303.us-east-1.redshift-serverless.amazonaws.com',
    database='dev',
    port=5439,
    user='bq2user',
    password='Sunrise@8785'  # Replace with your actual password or use a secure method
)

# Enable autocommit
conn.autocommit = True

# Create a cursor object
cursor = conn.cursor()

# SQL query to perform feature engineering
cursor.execute("DROP TABLE IF EXISTS bq2_data.latest_feature_engineered_data;")
sql_query = """
CREATE TABLE bq2_data.latest_feature_engineered_data AS
WITH carrier_counts AS (
    SELECT uniquecarrier, COUNT(*) AS carrier_count
    FROM bq2_data.joined_flights_full_weather
    GROUP BY uniquecarrier
),
total_count AS (
    SELECT COUNT(*) AS total_count FROM bq2_data.latest_joined_flights_full_weather
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
    FROM BQ2_data.latest_joined_flights_full_weather f
    JOIN carrier_counts c ON f.uniquecarrier = c.uniquecarrier
    CROSS JOIN total_count t
)
SELECT
    ROW_NUMBER() OVER () AS id,
    *
FROM feature_engineered;
"""
cursor.execute(sql_query)
print("SQL query executed successfully.")
cursor.close()
conn.close()


# In[10]:


import redshift_connector
# Establish the connection
conn = redshift_connector.connect(
    host='default-workgroup.820242926303.us-east-1.redshift-serverless.amazonaws.com',
    database='dev',
    port=5439,
    user='bq2user',
    password='Sunrise@8785'  # Replace with your actual password or use a secure method
)

# Enable autocommit
conn.autocommit = True

# Create a cursor object
cursor = conn.cursor()

unload_query = """
UNLOAD ('SELECT * FROM bq2_data.latest_feature_engineered_data')
TO 's3://airline-is459/data-source/BQ2/latest_feature_engineered_data/'
IAM_ROLE 'arn:aws:iam::820242926303:role/service-role/AmazonRedshift-CommandsAccessRole-20241017T010122'
FORMAT AS PARQUET
ALLOWOVERWRITE;
"""
# Execute the UNLOAD command
cursor.execute(unload_query)

# Close the cursor and connection
cursor.close()
conn.close()


# In[11]:


import dask.dataframe as dd

# Read the Parquet files from S3
ddf = dd.read_parquet('s3://airline-is459/data-source/BQ2/latest_feature_engineered_data/', storage_options={'anon': False})

# Proceed with data preprocessing and model training
ddf = ddf.dropna()


# In[12]:


from dask.distributed import Client
from dask_ml.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import boto3
import io
import joblib


# Initialize Dask client
client = Client(n_workers=4)

# Define features and target
X = ddf.drop(columns=['depdelay', 'arrdelay'])

# Add log transformations for selected features
X['distance_x_departure_hour'] = X['distance'] * X['departure_hour']
X['uniquecarrier_freq_x_distance'] = X['uniquecarrier_freq'] * X['distance']
X['log_distance'] = np.log1p(X['distance'])
X['log_uniquecarrier_freq'] = np.log1p(X['uniquecarrier_freq'])

# Categorical Weather Indicator: Severe Weather
X['severe_weather'] = ((X['origin_precipitation'] > 0.5) | (X['origin_wind_speed_10m'] > 20)).astype(int)

print(X.columns)

# Define target
y = ddf['depdelay'] + ddf['arrdelay']

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
)

# Convert training and validation data to Dask DMatrix
dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
dvalid = xgb.dask.DaskDMatrix(client, X_valid, y_valid)

# Define tuned model parameters without n_estimators, as it's controlled by num_boost_round
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.85,
    'colsample_bytree': 0.8
}

# Train the XGBoost model with early stopping
output = xgb.dask.train(
    client,
    params,
    dtrain,
    num_boost_round=400,  # Increased boosting rounds
    evals=[(dvalid, 'validation')],  # Validation set for early stopping
    early_stopping_rounds=50  # Stops if validation score doesn't improve for 50 rounds
)
booster = output['booster']

# Feature importance analysis
importance = booster.get_score(importance_type='weight')
importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

model_buffer = io.BytesIO()
joblib.dump(booster, model_buffer)
model_buffer.seek(0)

# Set up the S3 client and specify bucket and path
s3_client = boto3.client('s3')
s3_bucket_name = 'airline-is459'  # Replace with your S3 bucket name
s3_model_path = 'data-source/BQ2models/xgb_total_delay_model.joblib'  # Path in the bucket

# Upload buffer directly to S3
try:
    s3_client.upload_fileobj(model_buffer, s3_bucket_name, s3_model_path)
    print(f"Model uploaded to s3://{s3_bucket_name}/{s3_model_path}")
except Exception as e:
    print(f"Failed to upload model to S3: {e}")

# ---- Evaluation on the Test Set ----
# Prepare test data
dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)

# Perform predictions
y_pred = xgb.dask.predict(client, booster, dtest)

# Compute metrics
y_test_values = y_test.compute().copy()
y_pred_values = y_pred.compute().copy()
mse = mean_squared_error(y_test_values, y_pred_values)
mae = mean_absolute_error(y_test_values, y_pred_values)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_values, y_pred_values)

print("\nModel Performance on Test Data:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Display feature importance
print("\nFeature Importance:")
print(importance_df)

client.close()

# In[ ]:




