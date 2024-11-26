# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Set the environment variable to prevent matplotlib from trying to use any Xwindows backend
ENV MPLBACKEND=Agg

# Command to run your script
CMD ["python", "newBayesian.py"]
