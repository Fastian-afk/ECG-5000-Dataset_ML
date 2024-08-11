                                    ECG ANOMALY DETECTION USING TEMPORAL CONVOLUTIONAL NETWORK (TCN):
OVERVIEW:-

A) Objective: Detect anomalies in ECG (Electrocardiogram) data using machine learning techniques.
B) Key Features: Data preprocessing, TCN model training, and anomaly detection on ECG signals.

PREREQUISITES:-

1. Libraries Used:

NumPy, Pandas, Matplotlib for data manipulation and visualization.
Scikit-learn for data splitting and normalization.
Darts for time series modeling and anomaly detection.
TensorFlow, PyTorch Lightning for early stopping during model training.

PROJECT WORKFLOW:-

1. Loading and Merging the Dataset:
   
Data Sources: "train_data" and "test_data" loaded from text files.
Merged Data: Combined into a single dataset "merged_data" for streamlined processing.

2. Defining Normal and Anomalous Classes:
   
Class Definitions:
Normal: Label 1 (Normal ECG signals).
Anomalous: Any label other than 1.
Dataset Split: Created "normal_data" and "anomalous_data" subsets.

3. Data Splitting and Normalization:
   
Feature Selection: Extracted features into "X_normal", excluding labels.
Splitting: Used "train_test_split" for dividing the data into training, validation, and test sets.
Normalization: Applied "StandardScaler" to standardize features across datasets.

4. Converting Data to TimeSeries Format:
   
TimeSeries Conversion: Transformed data into TimeSeries objects, essential for time series modeling.
Ensuring Variability: Checked and modified data to ensure multiple samples in TimeSeries.

5. Implementing Early Stopping:
   
Purpose: Prevent overfitting by stopping training based on validation performance.
Configuration: Monitored "val_loss", stopping if no improvement by 0.05 for 5 epochs.

6. Training the TCN Model:
   
Model Configuration: Set input chunk length to 30 and output chunk length to 10.
Training: Trained on "series_train" (normal ECG data) with early stopping.
Model Saving: Saved the trained model for later use.

7. Anomaly Detection Model:
   
ForecastingAnomalyModel: Used the trained TCN model to detect anomalies by comparing predictions with actual values.
NormScorer: Calculated anomaly scores based on deviations.

8. Calculating Anomaly Scores and Threshold:
   
Threshold Determination: Set threshold as mean validation anomaly score plus three times the standard deviation.
Purpose: Classify ECG signals as anomalous or normal based on this threshold.

9. Evaluating on Test Data:
    
Anomaly Score Calculation: Applied the model to test data and calculated anomaly scores for both normal and anomalous signals.
Results: Printed anomaly scores for comparison.

10. Visualizing Results:
    
ECG Signal Plotting: Displayed ECG signals with the anomaly threshold, highlighting anomalous regions.
Anomaly Score Plotting: Showed anomaly scores for normal and anomalous data, with the threshold line.

KEY CONCEPTS:-

A) Time Series Modeling: Leveraged TCN for predicting future points in ECG time series.
B) Anomaly Detection: Identified unusual patterns by comparing predicted and actual ECG signals.
C) Early Stopping: Used to prevent model overfitting during training.

Contact
For questions or collaboration, please contact [Imaad Fazal or Ahmed Shaheer / imdufazal@gmail.com or ahmedshaheer605@gmail.com].
