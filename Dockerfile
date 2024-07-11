# Use a custom Python 3.12 image
# Use the official Python 3.12 image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy all the Python files and requirements.txt into the container
COPY . .

# Upgrade pip, setuptools, and wheel
RUN pip3 install --upgrade pip setuptools wheel

# Install the dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Command to run the main script when the container starts
CMD ["python", "./Main.py"]
