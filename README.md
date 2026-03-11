# Parkinson’s Disease Detection using Machine Learning and Federated Learning

## Project Overview
This project focuses on detecting **Parkinson’s Disease using voice measurements** with machine learning models. Parkinson’s disease affects speech patterns, and biomedical voice features can be used to classify whether a person has the disease.

The project compares multiple machine learning models and demonstrates a **Federated Learning simulation**, where multiple clients train models locally and share only model parameters instead of raw data. This helps preserve data privacy.

---

# Dataset

The dataset is taken from the **UCI Machine Learning Repository**.

Dataset Link:  
https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data

### Dataset Information
- Total Samples: **195**
- Total Features: **22 voice measurement features**
- Target Variable: **status**
  - `1` → Parkinson’s Disease
  - `0` → Healthy

### Example Features
- MDVP:Fo(Hz) – Average vocal fundamental frequency
- MDVP:Jitter(%) – Frequency variation
- MDVP:Shimmer – Amplitude variation
- NHR – Noise-to-Harmonics Ratio
- HNR – Harmonics-to-Noise Ratio
- RPDE – Recurrence Period Density Entropy
- DFA – Detrended Fluctuation Analysis
- PPE – Pitch Period Entropy

---

# Project Workflow

## 1. Data Loading
The dataset is loaded directly from the UCI repository using pandas.

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)
```

---

## 2. Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary column (`name`)
- Checked for missing values
- Checked for duplicate records
- Standardized features using **StandardScaler**
- Split dataset into **train (80%) and test (20%)**

```python
X = df.drop('status', axis=1)
y = df['status']
```

---

# Handling Class Imbalance

The dataset contains more Parkinson’s cases than healthy ones.  
To address this imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
```

---

# Machine Learning Models

Three classification models were implemented and compared.

## Logistic Regression
Baseline linear model used for binary classification.

Accuracy: **76.9%**

---

## Random Forest
Ensemble model using multiple decision trees.

Accuracy: **74.3%**

---

## Support Vector Machine (SVM)

SVM performed the best among the baseline models.

Accuracy: **82.0%**

---

# Hyperparameter Tuning

GridSearchCV was used to optimize SVM parameters.

Parameters tuned:

- C
- Kernel
- Gamma

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
```

### Best Parameters

```
C = 100
Kernel = rbf
Gamma = scale
```

### Final Model Performance

Accuracy: **89.7%**

---

# Model Evaluation

The final tuned SVM model was evaluated using:

- **Confusion Matrix**
- **ROC Curve**
- **AUC Score**
- **Permutation Feature Importance**

These evaluation techniques help analyze the classification performance and understand which voice features contribute most to predictions.

---

# Federated Learning Simulation

To demonstrate **privacy-preserving machine learning**, a federated learning simulation was implemented.

### Steps

1. Training data was split into **3 clients**
2. Each client trained a **local Logistic Regression model**
3. Model weights were shared instead of raw data
4. Parameters were aggregated using **FedAvg (Federated Averaging)**
5. Training was repeated for **5 rounds**

### Result

Global Model Accuracy: **89.7%**

This demonstrates how machine learning models can be trained collaboratively **without sharing sensitive data**.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn

---

# Project Structure

```
Parkinson-Disease-Detection
│
├── data
│   └── parkinsons.data
│
├── notebooks
│   └── model_training.ipynb
│
├── src
│   └── federated_learning.py
│
└── README.md
```

---

# Key Learnings

- Handling **imbalanced datasets using SMOTE**
- Training and comparing multiple ML models
- Hyperparameter tuning using **GridSearchCV**
- Model evaluation using **ROC curve and confusion matrix**
- Understanding **Federated Learning and privacy-preserving AI**

---

# Future Improvements

- Deep learning models for speech analysis
- Real-time prediction system
- Deployment using Flask or FastAPI
- Integration with healthcare applications
