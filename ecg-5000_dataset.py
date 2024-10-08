import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.ad import ForecastingAnomalyModel, NormScorer
from darts.models import TCNModel

# Loading the training dataset

train_data = pd.read_csv("ECG5000/ECG5000_TRAIN.txt", delim_whitespace=True, header=None)
print("Loaded Training Data:")
print(train_data)

# Loading the test dataset

test_data = pd.read_csv("ECG5000/ECG5000_TEST.txt", delim_whitespace=True, header=None)
print("Loaded Test Data:")
print(test_data)

# Display the shapes of the datasets

print(f"Shape of Training Data: {train_data.shape}")
print(f"Shape of Test Data: {test_data.shape}")

# Check for missing values (null values) in both datasets

print(f"Missing (null) values in Training Data:\n{train_data.isnull().sum()}")
print(f"Missing (null) values in Test Data:\n{test_data.isnull().sum()}")

# Display the first few entries of each dataset to understand their format

print("Sample Entries from Training Data:")
print(train_data.head())
print("Sample Entries from Test Data:")
print(test_data.head())

# Raise an alert for missing values

if train_data.isnull().sum().any():
    print("Warning: Training Data contains missing (null) values!")
else:
    print("No missing (null) values in Training Data.")

if test_data.isnull().sum().any():
    print("Warning: Test Data contains missing (null) values!")
else:
    print("No missing (null) values in Test Data.")

# Identify duplicate rows for both the datasets

train_duplicates_count = train_data.duplicated().sum()
test_duplicates_count = test_data.duplicated().sum()

print(f"Duplicate entries in the Training Data are: {train_duplicates_count}")
print(f"Duplicate entries in the Test Data are: {test_duplicates_count}")

# Raise an alert for duplicate rows

if train_duplicates_count > 0:
    print("Warning: Training Data has duplicate rows!")
else:
    print("No duplicate rows in Training Data.")

if test_duplicates_count > 0:
    print("Warning: Test Data has duplicate rows!")
else:
    print("No duplicate rows in Test Data.")

# Basic statistics for the datasets

print("General Statistics of Training Data are:")
print(train_data.describe())
print("General Statistics of Test Data are:")
print(test_data.describe())

#  Find and display count of each label in the datasets

print("Label Frequency in Training Data is:")
train_label_counts = train_data[0].value_counts()
print(train_label_counts)

print("Label Frequency in Test Data is:")
test_label_counts = test_data[0].value_counts()
print(test_label_counts)

# Determine which label represents "normal"

normal_class_label = 1  # Assumed normal class label

# Count of normal instances

normal_train_count = train_label_counts.get(normal_class_label, 0)
normal_test_count = test_label_counts.get(normal_class_label, 0)

print(f"Count of normal instances in Training Data is: {normal_train_count}")
print(f"Count of normal instances in Test Data is: {normal_test_count}")

# Combine the training and test datasets

merged_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
print("Shape of the Merged Data is:", merged_data.shape)

# Split the merged data into normal and anomalous subsets

normal_data = merged_data[merged_data[0] == normal_class_label]
anomalous_data = merged_data[merged_data[0] != normal_class_label]

# Plot one example from each category
plt.figure(figsize=(12, 6))

# Plot example from normal data
plt.subplot(1, 2, 1)
plt.plot(normal_data.iloc[0, 1:])
plt.title('Normal Instance')

# Plot example from anomalous data
plt.subplot(1, 2, 2)
plt.plot(anomalous_data.iloc[0, 1:])
plt.title('Anomalous Instance')

plt.tight_layout()
plt.show()

# Additional analysis of the combined dataset

# Label distribution in the merged data

print("Label Frequency in the Merged Data is:")
merged_label_counts = merged_data[0].value_counts()
print(merged_label_counts)

# 1. Visualizing the class distribution in merged data

plt.figure(figsize=(10, 5))
sns.countplot(x=merged_data[0])
plt.title("Class Distribution in Merged Data is:")
plt.show()

# 2. Plotting the Time Series samples from the merged data

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for i in range(3):
    axes[0, i].plot(merged_data.iloc[i, 1:])
    axes[0, i].set_title(f'Sample {i+1}')
    axes[1, i].plot(merged_data.iloc[i+3, 1:])
    axes[1, i].set_title(f'Sample {i+4}')
plt.tight_layout()
plt.show()

# 3. Heatmap of correlations in merged data

plt.figure(figsize=(15, 10))
sns.heatmap(merged_data.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap for the Merged Data")
plt.show()

# 4. Statistical summary of merged data features

print("Feature Statistics of Merged Data:")
print(merged_data.describe())

# 5. Boxplot for feature ranges and outliers

merged_data.iloc[:, 1:].boxplot(figsize=(20, 10))
plt.title("Boxplot of Features in the Merged Data")
plt.show()

sns.pairplot(merged_data.iloc[:, :5])
plt.show()