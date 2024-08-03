import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Combine normal and anomalous data
combined_data = pd.concat([normal_data, anomalous_data], axis=0).reset_index(drop=True)

# Split combined data into features and labels
X_combined = combined_data.iloc[:, 1:].values
y_combined = combined_data.iloc[:, 0].values

# Split the combined data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_combined, test_size=0.4, random_state=42, stratify=y_combined)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Display shapes of the splits
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")