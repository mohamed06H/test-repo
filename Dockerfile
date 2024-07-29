# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app/

# Ensure the WORKDIR and S3_BUCKET environment variables are passed into the container
ENV WORKDIR=/app/
ENV S3_BUCKET='raw'

# Command to run the application
CMD ["python", "app.py"]
