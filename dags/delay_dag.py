from airflow import DAG
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3

# Constants
S3_BUCKET = "airline-is459"
ENV_FILE_PATH = ".env"

# DAG configuration
dag = DAG(
    dag_id="model_training_delay_minutes",
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="Run data processing and model training on Redshift and S3 with MWAA",
    schedule_interval="@daily",
    start_date=datetime(2024, 11, 1),
    catchup=False,
)

# Function to download .env file from S3 and push variables to XCom
def load_env_vars_to_xcom(**kwargs):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=ENV_FILE_PATH)
    env_content = obj["Body"].read().decode("utf-8")
    env_vars = {}
    for line in env_content.splitlines():
        if line.strip() and not line.startswith("#"):
            key, value = line.split("=", 1)
            env_vars[key] = value
    
    # Push environment variables to XCom
    kwargs['ti'].xcom_push(key='env_vars', value=env_vars)

# Load environment variables task
load_env_task = PythonOperator(
    task_id="load_env_vars_to_xcom",
    python_callable=load_env_vars_to_xcom,
    provide_context=True,
    dag=dag,
)

# ECS task definition and cluster details
ecs_cluster = "is459_cluster"
ecs_task_definition = "model_training_delay_minutes"
security_group_id = "sg-0c960a1469146ca66"

# ECSOperator to execute the Python script, pulling env vars from XCom
def get_env_var(env_key, context):
    """Helper function to retrieve environment variables from XCom."""
    env_vars = context['ti'].xcom_pull(task_ids='load_env_vars_to_xcom', key='env_vars')
    return env_vars.get(env_key, '')

model_training_delay_minutes_task = EcsRunTaskOperator(
    task_id="model_training_delay_minutes_task",
    dag=dag,
    aws_conn_id="aws_default",
    cluster=ecs_cluster,
    task_definition=ecs_task_definition,
    launch_type="FARGATE",
    overrides={
        "containerOverrides": [
            {
                "name": "model_training_delay_minutes_container",
                'environmentFiles': [
                    {
                        'value': 'arn:aws:s3:::airline-is459/.env',
                        'type': 's3'
                    },
                ]
            }
        ],
    },
    network_configuration={
        "awsvpcConfiguration": {
            "subnets": [
                "subnet-061cd1991cf8a4a15",  # us-east-1e
                "subnet-0955541dc44ab7686",  # us-east-1a
                "subnet-0e795b80c4cdf155e",  # us-east-1f
                "subnet-0f3b93e22bd4b6eca",  # us-east-1d
                "subnet-04cb35894ea725c45",  # us-east-1b
                "subnet-075a0ba9e4c506bb7",  # us-east-1c
            ],
            "securityGroups": [security_group_id],
            "assignPublicIp": "ENABLED",
        }
    },
)

model_training_delay_type_task = EcsRunTaskOperator(
    task_id="model_training_delay_type_task",
    dag=dag,
    aws_conn_id="aws_default",
    cluster=ecs_cluster,
    task_definition=ecs_task_definition,
    launch_type="FARGATE",
    overrides={
        "containerOverrides": [
            {
                "name": "model_training_delay_type_container",
                'environmentFiles': [
                    {
                        'value': 'arn:aws:s3:::airline-is459/.env',
                        'type': 's3'
                    },
                ]
            }
        ],
    },
    network_configuration={
        "awsvpcConfiguration": {
            "subnets": [
                "subnet-061cd1991cf8a4a15",  # us-east-1e
                "subnet-0955541dc44ab7686",  # us-east-1a
                "subnet-0e795b80c4cdf155e",  # us-east-1f
                "subnet-0f3b93e22bd4b6eca",  # us-east-1d
                "subnet-04cb35894ea725c45",  # us-east-1b
                "subnet-075a0ba9e4c506bb7",  # us-east-1c
            ],
            "securityGroups": [security_group_id],
            "assignPublicIp": "ENABLED",
        }
    },
)

# Task dependencies
load_env_task >> [model_training_delay_minutes_task, model_training_delay_type_task]
