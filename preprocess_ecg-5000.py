import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Display counts of normal and anomalous data
print(f"Normal data count: {normal_data.shape[0]}")
print(f"Anomalous data count: {anomalous_data.shape[0]}")

# Split normal data into features and labels
X_normal = normal_data.iloc[:, 1:].values
y_normal = normal_data.iloc[:, 0].values

# Split anomalous data into features and labels
X_anomalous = anomalous_data.iloc[:, 1:].values
y_anomalous = anomalous_data.iloc[:, 0].values

# Split the normal data into training, validation, and test sets
X_train_normal, X_temp_normal, y_train_normal, y_temp_normal = train_test_split(X_normal, y_normal, test_size=0.4, random_state=42, stratify=y_normal)
X_val_normal, X_test_normal, y_val_normal, y_test_normal = train_test_split(X_temp_normal, y_temp_normal, test_size=0.5, random_state=42, stratify=y_temp_normal)

# Split the anomalous data into training, validation, and test sets
X_train_anomalous, X_temp_anomalous, y_train_anomalous, y_temp_anomalous = train_test_split(X_anomalous, y_anomalous, test_size=0.4, random_state=42, stratify=y_anomalous)
X_val_anomalous, X_test_anomalous, y_val_anomalous, y_test_anomalous = train_test_split(X_temp_anomalous, y_temp_anomalous, test_size=0.5, random_state=42, stratify=y_temp_anomalous)

# Normalize the normal data based on the training set
scaler_normal = StandardScaler()
X_train_normal = scaler_normal.fit_transform(X_train_normal)
X_val_normal = scaler_normal.transform(X_val_normal)
X_test_normal = scaler_normal.transform(X_test_normal)

# Normalize the anomalous data based on the training set
scaler_anomalous = StandardScaler()
X_train_anomalous = scaler_anomalous.fit_transform(X_train_anomalous)
X_val_anomalous = scaler_anomalous.transform(X_val_anomalous)
X_test_anomalous = scaler_anomalous.transform(X_test_anomalous)

# Display shapes of the splits
print(f"Normal Training data shape: {X_train_normal.shape}")
print(f"Normal Validation data shape: {X_val_normal.shape}")
print(f"Normal Test data shape: {X_test_normal.shape}")

print(f"Anomalous Training data shape: {X_train_anomalous.shape}")
print(f"Anomalous Validation data shape: {X_val_anomalous.shape}")
print(f"Anomalous Test data shape: {X_test_anomalous.shape}")

# Check the mean and standard deviation of the transformed data
print(f"Mean of the normal training data after normalization: {np.mean(X_train_normal, axis=0)}")
print(f"Standard deviation of the normal training data after normalization: {np.std(X_train_normal, axis=0)}")

print(f"Mean of the normal validation data after normalization: {np.mean(X_val_normal, axis=0)}")
print(f"Standard deviation of the normal validation data after normalization: {np.std(X_val_normal, axis=0)}")

print(f"Mean of the normal test data after normalization: {np.mean(X_test_normal, axis=0)}")
print(f"Standard deviation of the normal test data after normalization: {np.std(X_test_normal, axis=0)}")

print(f"Mean of the anomalous training data after normalization: {np.mean(X_train_anomalous, axis=0)}")
print(f"Standard deviation of the anomalous training data after normalization: {np.std(X_train_anomalous, axis=0)}")

print(f"Mean of the anomalous validation data after normalization: {np.mean(X_val_anomalous, axis=0)}")
print(f"Standard deviation of the anomalous validation data after normalization: {np.std(X_val_anomalous, axis=0)}")

print(f"Mean of the anomalous test data after normalization: {np.mean(X_test_anomalous, axis=0)}")
print(f"Standard deviation of the anomalous test data after normalization: {np.std(X_test_anomalous, axis=0)}")
