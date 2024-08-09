import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from darts import TimeSeries
from darts.models import TCNModel
from darts.ad import ForecastingAnomalyModel, NormScorer

# Load the dataset

train_data = pd.read_csv("ECG5000/ECG5000_TRAIN.txt", delim_whitespace=True, header=None)
test_data = pd.read_csv("ECG5000/ECG5000_TEST.txt", delim_whitespace=True, header=None)

# Merge the training and test datasets

merged_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

# Define normal and anomalous classes

normal_class_label = 1

# Split the data into normal and anomalous

normal_data = merged_data[merged_data[0] == normal_class_label]
anomalous_data = merged_data[merged_data[0] != normal_class_label]

# Split normal data into features and labels

X_normal = normal_data.iloc[:, 1:].values

# Split the normal data into training, validation, and test sets

X_train_normal, X_temp_normal = train_test_split(X_normal, test_size=0.4, random_state=42)
X_val_normal, X_test_normal = train_test_split(X_temp_normal, test_size=0.5, random_state=42)

# Normalize the normal data based on the training set

scaler_normal = StandardScaler()
X_train_normal = scaler_normal.fit_transform(X_train_normal)
X_val_normal = scaler_normal.transform(X_val_normal)
X_test_normal = scaler_normal.transform(X_test_normal)

# Convert the training data into Darts TimeSeries format

series_train = TimeSeries.from_values(X_train_normal)

# Ensure TimeSeries object is non-deterministic (has multiple samples)

if series_train.n_samples == 1:
    series_train = series_train.stack(series_train)


model = TCNModel(input_chunk_length=30, output_chunk_length=10, n_epochs=10, batch_size=8)
load_model = model.load("C:/Users/Hp/PycharmProjects/ECG-5000-Dataset_ML/TCN_Model.pth.tar")

model.fit(series_train)
model.save("C:/Users/Hp/PycharmProjects/ECG-5000-Dataset_ML/TCN_Model.pth.tar")
# Create an anomaly detection model using the trained forecasting model

anomaly_model = ForecastingAnomalyModel(model, scorer=NormScorer())

# Calculate anomaly scores on the validation data

series_val = TimeSeries.from_values(X_val_normal)
if series_val.n_samples == 1:
    series_val = series_val.stack(series_val)
anomaly_scores_val = anomaly_model.score(series_val)

# Decide the threshold based on validation data anomaly scores

mean_score = np.mean(anomaly_scores_val.values())
std_score = np.std(anomaly_scores_val.values())
threshold = mean_score + 3 * std_score
print(f"Anomaly detection threshold: {threshold}")

# Calculate anomaly scores on the test data (both normal and anomalous)

series_test_normal = TimeSeries.from_values(X_test_normal)
if series_test_normal.n_samples == 1:
    series_test_normal = series_test_normal.stack(series_test_normal)
anomaly_scores_test_normal = anomaly_model.score(series_test_normal)

# Print anomaly scores for normal test data

print("Anomaly Scores for Normal Test Data:")
print(anomaly_scores_test_normal.values())

# Plot the ECG graphs with predicted anomalies for normal test data

plt.figure(figsize=(12, 6))
plt.plot(X_test_normal[0], label='ECG Signal')
plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
plt.title('ECG Signal with Anomaly Threshold (Normal Data)')
plt.legend()
plt.show()

# Process the anomalous data

X_anomalous = anomalous_data.iloc[:, 1:].values

# Normalize the anomalous data

X_anomalous = scaler_normal.transform(X_anomalous)

# Calculate anomaly scores on the anomalous test data

series_anomalous = TimeSeries.from_values(X_anomalous)
if series_anomalous.n_samples == 1:
    series_anomalous = series_anomalous.stack(series_anomalous)
anomaly_scores_anomalous = anomaly_model.score(series_anomalous)

# Print anomaly scores for anomalous data

print("Anomaly Scores for Anomalous Test Data:")
print(anomaly_scores_anomalous.values())

# Plot the ECG graphs with predicted anomalies for anomalous data

plt.figure(figsize=(12, 6))
plt.plot(X_anomalous[0], label='ECG Signal')
plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
plt.title('ECG Signal with Anomaly Threshold (Anomalous Data)')
plt.legend()
plt.show()

# Additional analysis and visualization

# Plot anomaly scores for normal and anomalous data

plt.figure(figsize=(12, 6))
plt.plot(anomaly_scores_test_normal.values(), label='Normal Data Anomaly Scores')
plt.plot(anomaly_scores_anomalous.values(), label='Anomalous Data Anomaly Scores', color='red')
plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
plt.title('Anomaly Scores')
plt.legend()
plt.show()