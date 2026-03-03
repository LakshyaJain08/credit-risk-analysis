# 🏦 Credit Risk Prediction – LightGBM + Optuna

## 📌 Project Overview

This project builds an end-to-end **Credit Risk Classification Model** using the Home Credit Default Risk dataset. Read CreditEda.pdf for understanding the core objective

The objective is to identify high-risk loan applicants using:

- Advanced Feature Engineering
- Multicollinearity Reduction (VIF)
- SMOTE for Imbalanced Data
- LightGBM Gradient Boosting
- Optuna Hyperparameter Optimization (5-Fold Cross Validation)
- Precision-Recall Optimization
- Threshold Tuning
- Model Serialization for Deployment

---

## 🎯 Business Objective

Financial institutions must minimize loan defaults while maximizing approvals.

This model helps:

- Detect high-risk applicants
- Improve underwriting decisions
- Optimize risk-based pricing
- Enhance portfolio risk segmentation

---

## 📊 Dataset

Dataset: **Home Credit Default Risk**

Due to large file size, the dataset is NOT included in this repository.

Download from:

👉 https://www.kaggle.com/competitions/home-credit-default-risk/data

-After downloading, place the following file inside:
-data/raw/application_train.csv


---

## 🧠 Modeling Pipeline

### 1️⃣ Data Cleaning
- Remove high-missing columns (>60%)
- Median imputation for numerical features
- Mode/constant fill for categorical features

### 2️⃣ Feature Engineering
- Credit-to-Income Ratio
- Age Feature Creation

### 3️⃣ Multicollinearity Removal
- Variance Inflation Factor (VIF)
- Removed highly correlated numerical features

### 4️⃣ Encoding
- One-hot encoding
- Feature name sanitization (LightGBM safe)

### 5️⃣ Imbalance Handling
- SMOTE applied inside each CV fold

### 6️⃣ Model Training
- LightGBM Classifier
- 5-Fold Stratified Cross Validation
- Optuna Hyperparameter Tuning
- Early Stopping

### 7️⃣ Evaluation Metrics
- ROC AUC
- KS Statistic
- Precision-Recall Curve
- Average Precision (PR AUC)
- F1 Threshold Optimization

---

## 📈 Model Performance

- Cross-Validated AUC: ~0.69 (approx)
- KS Statistic: Calculated
- Precision-Recall Optimization Included
- Threshold tuning for business decisioning

---

## 🚀 How to Run

### 1️⃣ Clone Repository
-git clone https://github.com/LakshyaJain08/credit-risk-analysis.git

-cd credit-risk-analysis


### 2️⃣ Install Dependencies
-pip install -r requirements.txt


### 3️⃣ Place Dataset
-data/raw/application_train.csv


### 4️⃣ Run Model
-python main.py


---

## 📦 Output

After running:

- `credit_risk_model_optuna.pkl` → Trained Model
- `model_report.txt` → Full Evaluation Report

---

## ⚠️ Note on Large Files

- Raw dataset is excluded via `.gitignore`
- Model file is excluded
- Only source code is version controlled

---

## 🛠 Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- LightGBM
- Optuna
- imbalanced-learn
- statsmodels

---

## 👨‍💻 Author

Lakshya Jain  
Machine Learning & Data Science Enthusiast  

---

⭐ If you found this useful, consider starring the repo.