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

# Define start and end dates for some known anomalies
anomalies_day = {
    "NYC Marathon": ("2014-11-02 00:00", "2014-11-02 23:30"),
    "Thanksgiving": ("2014-11-27 00:00", "2014-11-27 23:30"),
    "Christmas": ("2014-12-24 00:00", "2014-12-25 23:30"),
    "New Years": ("2014-12-31 00:00", "2015-01-01 23:30"),
    "Snow Blizzard": ("2015-01-26 00:00", "2015-01-27 23:30"),
}
anomalies_day = {k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in anomalies_day.items()}

# Create a series with the binary anomaly flags
anomalies = pd.Series([0] * len(series_taxi), index=series_taxi.time_index)
for start, end in anomalies_day.values():
    anomalies.loc[(start <= anomalies.index) & (anomalies.index <= end)] = 1.0

series_taxi_anomalies = TimeSeries.from_series(anomalies)

# Plot the data and the anomalies
fig, ax = plt.subplots(figsize=(15, 5))
series_taxi.plot(label="Number of taxi passengers", linewidth=1, color="#6464ff")
(series_taxi_anomalies * 10000).plot(label="5 known anomalies", color="r", linewidth=1)
plt.legend()
plt.show()

# Only retain the data for the Thanksgiving anomaly
thanksgiving_start, thanksgiving_end = anomalies_day["Thanksgiving"]
series_taxi_thanksgiving = series_taxi.slice(thanksgiving_start - pd.Timedelta(days=30), thanksgiving_end + pd.Timedelta(days=30))

# Prepare the dataset for training
input_chunk_length = 30
output_chunk_length = 1
n_epochs = 40

# Split the series into training and validation sets
train, val = series_taxi_thanksgiving.split_before(0.8)

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

# Save the model
model.save("tcn_taxi_model.pth")

# Load the model
loaded_model = TCNModel.load("tcn_taxi_model.pth")

# Make predictions with the loaded model
pred_series = loaded_model.predict(len(val))

# Plot the predictions
plt.figure(figsize=(10, 6))
train.plot(label="Train")
val.plot(label="Validation")
pred_series.plot(label="Predictions")
plt.legend()
plt.show()

# Calculate and print the Mean Absolute Percentage Error
error = mape(val, pred_series)
print(f"Mean Absolute Percentage Error: {error:.2f}%")

# Detect anomalies based on prediction errors
threshold = 1.5 * np.std(train.values())  # Adjusted threshold
actual_anomalies = (val.values() > threshold).astype(int)
predicted_anomalies = (pred_series.values() > threshold).astype(int)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(actual_anomalies, predicted_anomalies)
precision = precision_score(actual_anomalies, predicted_anomalies)
recall = recall_score(actual_anomalies, predicted_anomalies)
f1 = f1_score(actual_anomalies, predicted_anomalies)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")