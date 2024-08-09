import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from darts import TimeSeries
from darts.models import TCNModel

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

# Create the TCN model
model = TCNModel(input_chunk_length=30, output_chunk_length=10, n_epochs=10, batch_size=8)

# Train the model
model.fit(series_train)

# Make predictions on the validation set (for example)
predictions = model.predict(len(X_val_normal))

# Plot the actual and predicted values
plt.plot(X_val_normal, label='Actual')
plt.plot(predictions.values(), label='Predicted')
plt.legend()
plt.show()