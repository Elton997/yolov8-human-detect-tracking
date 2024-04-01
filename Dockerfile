# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install pip requirements
RUN python -m pip install -r requirements.txt

# Run detect_and_track.py when the container launches
CMD ["python", "human_detect_and_tracking.py", "--video_path testing_data/input_video3.mp4", "--output_path testing_data/output_human3.mp4"]
