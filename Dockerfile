# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY scripts/model_training_delay_minutes.py /app/model_training_delay_minutes.py

# Command to run the Python script
CMD ["python3", "/app/model_training_delay_minutes.py"]
