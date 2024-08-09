import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TCNModel
from darts.metrics import mape
from darts.datasets import TaxiNewYorkDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data
series_taxi = TaxiNewYorkDataset().load()

# Define start and end dates for a custom anomaly period (replace with your desired dates)
custom_anomaly_start = pd.Timestamp("2015-02-15")  # Adjust as needed
custom_anomaly_end = pd.Timestamp("2015-02-22")  # Adjust as needed

# Create a series with a binary anomaly flag for the custom period
anomalies = pd.Series([0] * len(series_taxi), index=series_taxi.time_index)
anomalies.loc[(custom_anomaly_start <= anomalies.index) & (anomalies.index <= custom_anomaly_end)] = 1.0

series_taxi_anomalies = TimeSeries.from_series(anomalies)

# Plot the data and custom anomaly flag
fig, ax = plt.subplots(figsize=(15, 5))
series_taxi.plot(label="Number of taxi passengers", linewidth=1, color="#6464ff")
(series_taxi_anomalies * 10000).plot(label="Custom anomaly period", color="r", linewidth=1)
plt.legend()
plt.show()

# Retain data for a broader period around the custom anomaly
buffer_days = 15  # Adjust buffer days as needed
series_taxi_custom = series_taxi.slice(custom_anomaly_start - pd.Timedelta(days=buffer_days),
                                       custom_anomaly_end + pd.Timedelta(days=buffer_days))

# Training parameters (adjust as needed)
input_chunk_length = 30
output_chunk_length = 1
n_epochs = 1000

# Split the series into training and validation sets (consider using a time-based split)
train, val = series_taxi_custom.split_before(0.8)

# Create and train the TCN model
model = TCNModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    kernel_size=2,
    num_filters=16,
    n_epochs=n_epochs,
    dropout=0.2,
    random_state=42
)
model.fit(train)

# Save the model (optional)
model.save("tcn_taxi_custom_model.pth")

# Load the model (if saved previously)
# loaded_model = TCNModel.load("tcn_taxi_custom_model.pth")

# Make predictions
pred_series = model.predict(len(val))

# Plot the predictions
plt.figure(figsize=(10, 6))
train.plot(label="Train")
val.plot(label="Validation")
pred_series.plot(label="Predictions")
plt.legend()
plt.show()

# Calculate MAPE
error = mape(val, pred_series)
print(f"Mean Absolute Percentage Error: {error:.2f}%")

# Anomaly detection based on prediction errors
threshold = 1.5 * np.std(train.values())  # Adjusted threshold
actual_anomalies = (val.values() > threshold).astype(int)
predicted_anomalies = (pred_series.values() > threshold).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(actual_anomalies, predicted_anomalies)
precision = precision_score(actual_anomalies, predicted_anomalies)
recall = recall_score(actual_anomalies, predicted_anomalies)
f1 = f1_score(actual_anomalies, predicted_anomalies)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")