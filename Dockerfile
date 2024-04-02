# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install pip requirements
RUN python -m pip install -r requirements.txt

# Command to run the Python script with CMD arguments
CMD ["python", "human_detect_and_tracking.py"]