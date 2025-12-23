# ❤️ ECG Anomaly Detection using Temporal Convolutional Networks (TCN)
**Time-Series Anomaly Detection for Cardiac Signals**

<p align="center">
  <img src="https://img.shields.io/badge/Time%20Series-ECG-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-TCN-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Anomaly%20Detection-red?style=for-the-badge"/>
</p>

---

## 📌 Overview
This project focuses on **detecting anomalies in ECG (Electrocardiogram) signals** using a **Temporal Convolutional Network (TCN)** trained on *normal* cardiac rhythms.

Instead of treating anomaly detection as a classification task, the model **learns normal ECG behavior** and flags deviations using **forecasting-based anomaly scoring**, a robust and scalable approach widely used in real-world monitoring systems.

---

## 🎯 Objective
- Detect abnormal ECG patterns that may indicate cardiac irregularities  
- Leverage **deep time-series modeling** instead of rule-based detection  
- Apply **statistical thresholding** for interpretable anomaly decisions  

---

## 🔬 Key Features
- ⏱ Time-series forecasting using **TCN**
- 🧠 Training exclusively on **normal ECG signals**
- 📊 Statistical anomaly scoring with dynamic thresholding
- 🛑 Early stopping to prevent overfitting
- 📈 Visualization of ECG signals and anomaly scores

---

## 🛠 Tech Stack
<p align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="38"/>
</p>

**Libraries & Frameworks**
- NumPy, Pandas — data processing  
- Matplotlib — visualization  
- Scikit-learn — splitting & normalization  
- **Darts** — time-series modeling & anomaly detection  
- PyTorch Lightning — training utilities & early stopping  

---

## 📂 Project Workflow

### 1️⃣ Data Loading & Merging
- ECG-5000 dataset loaded from `train_data` and `test_data`
- Combined into a unified dataset for streamlined processing

---

### 2️⃣ Normal vs Anomalous Definition
- **Normal ECG** → Label `1`
- **Anomalous ECG** → Any label ≠ `1`
- Dataset split into:
  - `normal_data`
  - `anomalous_data`

---

### 3️⃣ Data Splitting & Normalization
- Features extracted (labels excluded)
- Train / validation / test split using `train_test_split`
- Standardization via `StandardScaler`

---

### 4️⃣ TimeSeries Conversion
- Data converted into **Darts `TimeSeries` objects**
- Ensured sufficient temporal variability for model learning

---

### 5️⃣ Early Stopping Strategy
- Monitored: `val_loss`
- Patience: 5 epochs  
- Minimum improvement threshold: `0.05`
- Prevents overfitting and unnecessary training

---

### 6️⃣ TCN Model Training
- Input chunk length: `30`
- Output chunk length: `10`
- Trained only on **normal ECG signals**
- Final model saved for reuse

---

### 7️⃣ Anomaly Detection
- Used **ForecastingAnomalyModel**
- Compared predicted ECG values with actual signals
- **NormScorer** used to compute anomaly scores

---

### 8️⃣ Threshold Selection
- Threshold =  
  **mean(validation scores) + 3 × std(validation scores)**
- Enables statistically interpretable anomaly detection

---

### 9️⃣ Evaluation on Test Data
- Anomaly scores computed for:
  - Normal ECG samples
  - Anomalous ECG samples
- Scores printed for direct comparison

---

### 🔟 Visualization
- 📉 ECG signal plots with anomaly regions highlighted
- 📊 Anomaly score plots with threshold overlay

---

## 📊 Why This Approach Works
- Learns **normal cardiac behavior** instead of memorizing anomalies
- Robust to unseen anomaly patterns
- Aligns with **industry-grade monitoring systems**
- Scales well to real-time ECG streams

---

## 👨‍💻 Authors
**Imaad Fazal**  
**Ahmed Shaheer**

📧 Emails:  
- imdufazal@gmail.com  
- ahmedshaheer605@gmail.com  

---

## 📜 License
This project is released under the **MIT License**.
