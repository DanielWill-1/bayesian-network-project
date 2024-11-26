import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
# Ensure that the 'weatherAUS.csv' file is in your working directory
data = pd.read_csv('C:/Users/danie/OneDrive/Documents/GitHub/bayesian-network-project/weatherAUS.csv')

# Data Preprocessing
# Select relevant columns
data = data[['Location', 'MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm', 'Rainfall']]

# Handle missing values by dropping rows with NaNs
data.dropna(inplace=True)

# Average MinTemp and MaxTemp to get a general Temperature
data['Temperature'] = data[['MinTemp', 'MaxTemp']].mean(axis=1)

# Average Humidity9am and Humidity3pm to get a general Humidity
data['Humidity'] = data[['Humidity9am', 'Humidity3pm']].mean(axis=1)

# Use Rainfall as Precipitation
data['Precipitation'] = data['Rainfall']

# Select the new columns
data = data[['Location', 'Temperature', 'Humidity', 'Precipitation']]

# Discretize continuous variables using quantiles
from sklearn.preprocessing import KBinsDiscretizer

continuous_vars = ['Temperature', 'Humidity', 'Precipitation']

est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
data[continuous_vars] = est.fit_transform(data[continuous_vars])

data[continuous_vars] = data[continuous_vars].astype(int)

from sklearn.preprocessing import LabelEncoder

location_encoder = LabelEncoder()
data['Location_encoded'] = location_encoder.fit_transform(data['Location'])

# Update data to use encoded location
data_model = data[['Location_encoded', 'Temperature', 'Humidity', 'Precipitation']]


hc = HillClimbSearch(data_model)  # Remove scoring_method from here
best_model = hc.estimate(scoring_method=BicScore(data_model))  # Pass scoring_method to estimate()
# Create the Bayesian Model with the learned structure
model = BayesianModel(best_model.edges())

# Estimate the Conditional Probability Distributions (CPDs)
model.fit(data_model, estimator=MaximumLikelihoodEstimator)

# Perform inference with the model
infer = VariableElimination(model)

# Allow the user to input a location
user_location = input("Enter a location (e.g., Sydney): ")

# Check if the entered location exists in the dataset
if user_location not in location_encoder.classes_:
    print(f"Location '{user_location}' not found in the dataset.")
    print("Available locations are:", list(location_encoder.classes_))
else:
    # Get the encoded value of the location
    encoded_location = location_encoder.transform([user_location])[0]

    # Optionally, you can ask the user to input Temperature or use default
    # For simplicity, let's ask for Temperature bin
    temperature_bin = input("Enter Temperature bin (0: Low, 1: Medium, 2: High, leave blank if unknown): ")
    if temperature_bin == '':
        # If temperature is unknown, we don't set it in evidence
        evidence = {'Location_encoded': encoded_location}
    else:
        try:
            temperature_bin = int(temperature_bin)
            if temperature_bin not in [0, 1, 2]:
                raise ValueError
            evidence = {'Location_encoded': encoded_location, 'Temperature': temperature_bin}
        except ValueError:
            print("Invalid Temperature bin. Please enter 0, 1, or 2.")
            exit()

    # Predict Humidity
    q_humidity = infer.query(variables=['Humidity'], evidence=evidence)

    # Predict Precipitation
    q_precipitation = infer.query(variables=['Precipitation'], evidence=evidence)

    # Map the bins back to approximate ranges
    # Get the bin edges for Precipitation
    precipitation_bin_edges = est.bin_edges_[continuous_vars.index('Precipitation')]

    # Calculate expected rainfall amount
    expected_rainfall = 0
    for bin_idx in range(len(q_precipitation.values)):
        # Calculate the midpoint of the bin
        bin_start = precipitation_bin_edges[bin_idx]
        bin_end = precipitation_bin_edges[bin_idx + 1]
        bin_midpoint = (bin_start + bin_end) / 2
        # Multiply by the probability
        expected_rainfall += bin_midpoint * q_precipitation.values[bin_idx]

    # Output the predictions
    print("\nPredicted Humidity given evidence:")
    for idx, prob in enumerate(q_humidity.values):
        print(f"Humidity bin {idx} (approx. {est.bin_edges_[continuous_vars.index('Humidity')][idx]:.1f}% - {est.bin_edges_[continuous_vars.index('Humidity')][idx+1]:.1f}%): Probability = {prob:.4f}")

    print("\nPredicted Precipitation given evidence:")
    for idx, prob in enumerate(q_precipitation.values):
        print(f"Precipitation bin {idx} (approx. {est.bin_edges_[continuous_vars.index('Precipitation')][idx]:.1f}mm - {est.bin_edges_[continuous_vars.index('Precipitation')][idx+1]:.1f}mm): Probability = {prob:.4f}")

    print(f"\nExpected Rainfall Amount: {expected_rainfall:.2f} mm")
plt.figure(figsize=(8,6))
sns.histplot(data['Temperature'], bins=30, kde=True)
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()

# Correlation Matrix
plt.figure(figsize=(8,6))
# Correlation Matrix
plt.figure(figsize=(8,6))
# Calculate correlation on numerical columns only
corr = data[['Temperature', 'Humidity', 'Precipitation']].corr()  # Select only numerical columns
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Boxplot of Humidity by Location
plt.figure(figsize=(12,6))
sns.boxplot(x='Location', y='Humidity', data=data)
plt.title('Humidity by Location')
plt.xlabel('Location')
plt.ylabel('Humidity')
plt.xticks(rotation=90)
plt.show()