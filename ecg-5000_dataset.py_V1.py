import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.ad import ForecastingAnomalyModel, NormScorer
from darts.models import TCNModel


def load_and_inspect_data(train_path, test_path):
    """
    Load and inspect the training and test datasets.

    Parameters:
    train_path (str): Path to the training data file.
    test_path (str): Path to the test data file.

    Returns:
    pd.DataFrame, pd.DataFrame: Training and test data as pandas DataFrames.
    """
    # Loading the datasets
    train_data = pd.read_csv(train_path, delim_whitespace=True, header=None)
    test_data = pd.read_csv(test_path, delim_whitespace=True, header=None)

    # Display the shapes and check for missing values
    print(f"Shape of Training Data: {train_data.shape}")
    print(f"Shape of Test Data: {test_data.shape}")

    print(f"Missing (null) values in Training Data:\n{train_data.isnull().sum()}")
    print(f"Missing (null) values in Test Data:\n{test_data.isnull().sum()}")

    return train_data, test_data


def check_duplicates_and_stats(data, name="Dataset"):
    """
    Check for duplicates and display basic statistics for the dataset.

    Parameters:
    data (pd.DataFrame): The dataset to analyze.
    name (str): Name of the dataset for display purposes.
    """
    duplicates_count = data.duplicated().sum()
    if duplicates_count > 0:
        print(f"Warning: {name} has {duplicates_count} duplicate rows!")
    else:
        print(f"No duplicate rows in {name}.")

    print(f"General Statistics of {name}:")
    print(data.describe())


def visualize_data_distribution(data, title):
    """
    Visualize the label distribution in the dataset.

    Parameters:
    data (pd.DataFrame): The dataset with labels in the first column.
    title (str): Title for the visualization.
    """
    plt.figure(figsize=(10, 5))
    sns.countplot(x=data[0])
    plt.title(title)
    plt.show()


def plot_sample_instances(normal_data, anomalous_data):
    """
    Plot examples of normal and anomalous instances.

    Parameters:
    normal_data (pd.DataFrame): Subset of normal data.
    anomalous_data (pd.DataFrame): Subset of anomalous data.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(normal_data.iloc[0, 1:])
    plt.title('Normal Instance')

    plt.subplot(1, 2, 2)
    plt.plot(anomalous_data.iloc[0, 1:])
    plt.title('Anomalous Instance')

    plt.tight_layout()
    plt.show()


def additional_analysis(merged_data):
    """
    Perform additional analysis on the merged dataset.

    Parameters:
    merged_data (pd.DataFrame): The combined dataset.
    """
    # Label distribution visualization
    visualize_data_distribution(merged_data, "Class Distribution in Merged Data")

    # Time Series sample plots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for i in range(3):
        axes[0, i].plot(merged_data.iloc[i, 1:])
        axes[0, i].set_title(f'Sample {i + 1}')
        axes[1, i].plot(merged_data.iloc[i + 3, 1:])
        axes[1, i].set_title(f'Sample {i + 4}')
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(merged_data.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap for the Merged Data")
    plt.show()

    # Feature boxplot
    merged_data.iloc[:, 1:].boxplot(figsize=(20, 10))
    plt.title("Boxplot of Features in the Merged Data")
    plt.show()

    # Pairplot for first few features
    sns.pairplot(merged_data.iloc[:, :5])
    plt.show()


# Paths to the datasets
train_path = "ECG5000/ECG5000_TRAIN.txt"
test_path = "ECG5000/ECG5000_TEST.txt"

# Load and inspect the data
train_data, test_data = load_and_inspect_data(train_path, test_path)

# Check for duplicates and display statistics
check_duplicates_and_stats(train_data, "Training Data")
check_duplicates_and_stats(test_data, "Test Data")

# Identify the normal class label
normal_class_label = 1

# Combine the datasets
merged_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
print("Shape of the Merged Data:", merged_data.shape)

# Split the data into normal and anomalous subsets
normal_data = merged_data[merged_data[0] == normal_class_label]
anomalous_data = merged_data[merged_data[0] != normal_class_label]

# Plot sample instances
plot_sample_instances(normal_data, anomalous_data)

# Additional analysis of the merged data
additional_analysis(merged_data)