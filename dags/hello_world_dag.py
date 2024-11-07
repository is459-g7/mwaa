from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Define the default_args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
with DAG(
    'hello_world_dag',
    default_args=default_args,
    description='A simple Hello World DAG for MWAA',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    def print_hello():
        print("Hello, World!")

    # Define a PythonOperator to print "Hello, World!"
    hello_task = PythonOperator(
        task_id='hello_task',
        python_callable=print_hello,
    )
