from airflow import DAG
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

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

# Python function to retrieve environment variables from MWAA
def load_env_vars():
    for key, value in os.environ.items():
        print(f"{key}: {value}")

# Load environment variables task
load_env_task = PythonOperator(
    task_id="load_env_vars",
    python_callable=load_env_vars,
    dag=dag,
)

# ECS task definition name and cluster details
ecs_cluster = "is459_cluster"
ecs_task_definition = "model_training_delay_minutes"
subnet_id = "subnet-061cd1991cf8a4a15"
security_group_id = "sg-0c960a1469146ca66"

# ECSOperator to execute the Python script
data_processing_task = EcsRunTaskOperator(
    task_id="model_training_delay_minutes_task",
    dag=dag,
    aws_conn_id="aws_default",
    cluster=ecs_cluster,
    task_definition=ecs_task_definition,
    launch_type="FARGATE",
    overrides={
        "containerOverrides": [
            {
                "name": "data_processing_container",
                "command": ["python3", "/opt/scripts/model_training_delay_minutes.py"],  # Update path to your Python file
                "environment": [
                    {"name": "REDSHIFT_HOST", "value": os.getenv("REDSHIFT_HOST")},
                    {"name": "REDSHIFT_DB", "value": os.getenv("REDSHIFT_DB")},
                    {"name": "REDSHIFT_PORT", "value": os.getenv("REDSHIFT_PORT")},
                    {"name": "REDSHIFT_USER", "value": os.getenv("REDSHIFT_USER")},
                    {"name": "REDSHIFT_PASSWORD", "value": os.getenv("REDSHIFT_PASSWORD")},
                    {"name": "S3_BUCKET", "value": os.getenv("S3_BUCKET")},
                    {"name": "IAM_ROLE_ARN", "value": os.getenv("IAM_ROLE_ARN")},
                ],
            }
        ],
    },
    network_configuration={
        "awsvpcConfiguration": {
            "subnets": [subnet_id],
            "securityGroups": [security_group_id],
            "assignPublicIp": "ENABLED",
        }
    },
)

# Task dependencies
load_env_task >> data_processing_task
