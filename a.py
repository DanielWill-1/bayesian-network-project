# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Imports for PyTorch and model evaluation
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Imports for Prometheus monitoring
from prometheus_client import start_http_server, Summary, Counter, Gauge
import time

# Imports for MLflow
import mlflow
import mlflow.pytorch
import subprocess  # To start MLflow UI programmatically
import sys         # To get the path to the Python executable
import os          # For path manipulations

if getattr(sys, 'frozen', False):
    # If the application is running as a bundle (e.g., with PyInstaller)
    script_dir = sys._MEIPASS
else:
    # If running in a normal Python environment
    script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data file
data_path = os.path.join(script_dir, 'weatherAUS.csv')

# Validate the existence of weatherAUS.csv
if not os.path.isfile(data_path):
    sys.exit(f"Error: Data file '{data_path}' not found. Ensure the file is in the correct location.")

# Set the tracking URI for MLflow (adjust the path as needed)
mlruns_path = os.path.join(script_dir, 'mlruns')
mlflow.set_tracking_uri(f'file:///{mlruns_path}')

# Ensure the 'mlruns' directory exists
if not os.path.exists(mlruns_path):
    os.makedirs(mlruns_path)

# Start the MLflow UI programmatically
mlflow_ui_process = subprocess.Popen([
    sys.executable, '-m', 'mlflow', 'ui',
    '--backend-store-uri', f'file:///{mlruns_path}',
    '--port', '5000'
])
print("MLflow UI started at http://localhost:5000")

# Load the dataset
data = pd.read_csv(data_path)

# Data Preprocessing
# ==================
data = data[['Location', 'MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm', 'Rainfall']]
data.dropna(inplace=True)
data['Temperature'] = data[['MinTemp', 'MaxTemp']].mean(axis=1)
data['Humidity'] = data[['Humidity9am', 'Humidity3pm']].mean(axis=1)
data['Precipitation'] = data['Rainfall']
data = data[['Location', 'Temperature', 'Humidity', 'Precipitation']]

# Discretize continuous variables
from sklearn.preprocessing import KBinsDiscretizer
continuous_vars = ['Temperature', 'Humidity', 'Precipitation']
est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
data[continuous_vars] = est.fit_transform(data[continuous_vars])
data[continuous_vars] = data[continuous_vars].astype(int)

# Encode categorical variables
location_encoder = LabelEncoder()
data['Location_encoded'] = location_encoder.fit_transform(data['Location'])
data_model = data[['Location_encoded', 'Temperature', 'Humidity', 'Precipitation']]

# Bayesian Network Structure Learning
# ===================================
hc = HillClimbSearch(data_model)
best_model = hc.estimate(scoring_method=BicScore(data_model))
model = BayesianModel(best_model.edges())
model.fit(data_model, estimator=MaximumLikelihoodEstimator)
infer = VariableElimination(model)

# User Interaction and Predictions
# ================================
user_location = input("Enter a location (e.g., Sydney): ")

# Check if the entered location exists in the dataset
if user_location not in location_encoder.classes_:
    print(f"Location '{user_location}' not found in the dataset.")
    print("Available locations are:", list(location_encoder.classes_))
    sys.exit("Exiting due to invalid location.")

temperature_bin = input("Enter Temperature bin (0: Low, 1: Medium, 2: High, leave blank if unknown): ")
if temperature_bin == '':
    evidence = {'Location_encoded': location_encoder.transform([user_location])[0]}
else:
    try:
        temperature_bin = int(temperature_bin)
        if temperature_bin not in [0, 1, 2]:
            raise ValueError
        evidence = {
            'Location_encoded': location_encoder.transform([user_location])[0],
            'Temperature': temperature_bin
        }
    except ValueError:
        sys.exit("Invalid Temperature bin. Please enter 0, 1, or 2.")

# Predict Humidity and Precipitation
q_humidity = infer.query(variables=['Humidity'], evidence=evidence)
q_precipitation = infer.query(variables=['Precipitation'], evidence=evidence)

# Calculate expected rainfall amount
precipitation_bin_edges = est.bin_edges_[continuous_vars.index('Precipitation')]
expected_rainfall = sum(
    ((precipitation_bin_edges[i] + precipitation_bin_edges[i+1]) / 2) * q_precipitation.values[i]
    for i in range(len(q_precipitation.values))
)

# Output Predictions
print("\nPredicted Humidity given evidence:")
for idx, prob in enumerate(q_humidity.values):
    print(f"Humidity bin {idx} (approx. {est.bin_edges_[continuous_vars.index('Humidity')][idx]:.1f}% - {est.bin_edges_[continuous_vars.index('Humidity')][idx+1]:.1f}%): Probability = {prob:.4f}")

print("\nPredicted Precipitation given evidence:")
for idx, prob in enumerate(q_precipitation.values):
    print(f"Precipitation bin {idx} (approx. {est.bin_edges_[continuous_vars.index('Precipitation')][idx]:.1f}mm - {est.bin_edges_[continuous_vars.index('Precipitation')][idx+1]:.1f}mm): Probability = {prob:.4f}")

print(f"\nExpected Rainfall Amount: {expected_rainfall:.2f} mm")

# Validate model file existence
humidity_model_path = 'model_humidity.pth'
precipitation_model_path = 'model_precipitation.pth'

if not os.path.exists(humidity_model_path) or not os.path.exists(precipitation_model_path):
    sys.exit(f"Error: Required model files ('{humidity_model_path}', '{precipitation_model_path}') are missing.")

# Prometheus Monitoring Adjustments
time.sleep(10)  # Wait before shutting down the Prometheus server
print("Shutting down Prometheus metrics server.")

# Terminate MLflow UI
mlflow_ui_process.terminate()
print("MLflow UI has been stopped.")
