#!/usr/bin/env python
# coding: utf-8

# # CREDIT RISK ANALYTICS SYSTEM

# ## PART 1 – STRUCTURAL AUDIT

# In[1]:


"""
Objective:

Understand portfolio risk profile, dataset size, imbalance severity, and missing data complexity.

Description:

-This stage evaluates:
-Dataset dimensionality
-Default distribution
-Missing value ratio

Data readiness for preprocessing
"""


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# CREATE PROJECT STRUCTURE

for folder in ["outputs/figures",
               "outputs/reports",
               "data/processed",
               "docs"]:
    Path(folder).mkdir(parents=True, exist_ok=True)


# In[3]:


# PART 1 – STRUCTURAL AUDIT

app = pd.read_csv("data/raw/application_data.csv")

total_apps = app.shape[0]
total_features = app.shape[1]
default_rate = app["TARGET"].mean()*100
missing_ratio = app.isnull().sum().sum() / (app.shape[0]*app.shape[1]) * 100

plt.figure(figsize=(6,4))
sns.countplot(x="TARGET", data=app)
plt.title("Target Distribution")
plt.savefig("outputs/figures/01_target_distribution.png", dpi=300)
plt.show()
plt.close()

report1 = f"""
PART 1 – STRUCTURAL AUDIT REPORT

Total Applications: {total_apps}
Total Features: {total_features}
Default Rate: {default_rate:.2f}%
Missing Ratio: {missing_ratio:.2f}%

Interpretation:
Portfolio shows moderate imbalance and missing complexity.
"""

with open("outputs/reports/01_structural_audit_report.txt","w") as f:
    f.write(report1)


# ## PART 2 – DATA CLEANING

# In[4]:


"""
Objective:

-Improve dataset quality and remove unusable features.

Description:

-Drops high-missing columns
-Imputes numerical & categorical values
-Improves data completeness
"""


# In[5]:


df = app.copy()

missing_before = df.isnull().sum().sum()

df = df.loc[:, df.isnull().mean() < 0.6]

for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

missing_after = df.isnull().sum().sum()
quality_score = (1 - missing_after/(df.shape[0]*df.shape[1]))*100

df.to_csv("data/processed/application_cleaned.csv", index=False)

report2 = f"""
PART 2 – DATA CLEANING REPORT

Missing Before: {missing_before}
Missing After: {missing_after}
Data Quality Score: {quality_score:.2f}%

Dataset ready for EDA.
"""

with open("outputs/reports/02_cleaning_report.txt","w") as f:
    f.write(report2)


# ## PART 3 – FULL EDA

# In[6]:


"""
Objective:
Perform deep exploratory analysis using univariate, 
bivariate and multivariate techniques to uncover 
financial and demographic risk drivers.

Description:
This stage evaluates:
- Distribution shape & skewness
- Outlier structure
- Default rate by segments
- Financial leverage risk
- Correlation strength
- Multicollinearity assessment
"""


# In[7]:


import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ### UNIVARIATE ANALYSIS

# In[8]:


num_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT",
            "AMT_ANNUITY", "DAYS_BIRTH"]

univariate_summary = []

plt.figure(figsize=(14,10))
for i, col in enumerate(num_cols,1):
    plt.subplot(2,2,i)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f"{col} Distribution")
    
    skewness = df[col].skew()
    kurtosis = df[col].kurt()
    univariate_summary.append(
        f"{col} → Skew: {skewness:.2f}, Kurtosis: {kurtosis:.2f}"
    )

plt.tight_layout()
plt.show()
plt.savefig("outputs/figures/06_univariate_numerical_advanced.png", dpi=300)
plt.close()

# Categorical frequency %
cat_cols = ["NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "CODE_GENDER"]

cat_summary = []
for col in cat_cols:
    freq_pct = (df[col].value_counts(normalize=True)*100).round(2)
    cat_summary.append(f"\n{col} Distribution (%):\n{freq_pct.to_string()}")


# ### BIVARIATE ANALYSIS

# In[9]:


# Numerical vs TARGET – Mean comparison

bivariate_summary = []

for col in num_cols:
    mean_default = df[df["TARGET"]==1][col].mean()
    mean_non_default = df[df["TARGET"]==0][col].mean()
    
    bivariate_summary.append(
        f"{col} → Default Mean: {mean_default:.2f}, "
        f"Non-Default Mean: {mean_non_default:.2f}"
    )

# Credit-Income Risk Lift
df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
df["CIR_Q"] = pd.qcut(df["CREDIT_INCOME_RATIO"], 4)

risk_lift = df.groupby("CIR_Q")["TARGET"].mean()

plt.figure(figsize=(6,4))
risk_lift.plot(kind="bar")
plt.title("Default Rate by Credit-Income Quartile")
plt.show()
plt.savefig("outputs/figures/07_bivariate_risk_lift.png", dpi=300)
plt.close()

# Age segmentation
df["AGE"] = abs(df["DAYS_BIRTH"])/365
df["AGE_GROUP"] = pd.cut(df["AGE"], bins=[20,30,40,50,60,100])
age_default = df.groupby("AGE_GROUP")["TARGET"].mean()


# ### MULTIVARIATE ANALYSIS

# In[10]:


# Correlation with TARGET

corr_matrix = df.select_dtypes(include=np.number).corr()
corr_target = corr_matrix["TARGET"].sort_values(ascending=False)

plt.figure(figsize=(52,50))
sns.heatmap(corr_matrix, cmap="coolwarm", fmt='.2f',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.5})
plt.title("Correlation Matrix")
plt.show()
plt.savefig("outputs/figures/08_multivariate_correlation.png", dpi=300)
plt.close()

# VIF Calculation (multicollinearity check)
vif_data = pd.DataFrame()
X_vif = df[num_cols].dropna()

vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

vif_data.to_csv("docs/vif_report.csv", index=False)


# In[11]:


# VIF Analysis

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

num_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT",
            "AMT_ANNUITY", "DAYS_BIRTH"]

X_vif = df[num_cols].dropna()

vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

# Convert to R² %
vif_data["R2_Percentage"] = (1 - (1 / vif_data["VIF"])) * 100

print(vif_data)


# In[12]:


# Since VIF% >5 we have to remove high value VIF

def remove_high_vif(data, thresh=5.0):
    
    while True:
        
        vif = pd.DataFrame()
        vif['Feature'] = data.columns
        vif['VIF'] = [
            variance_inflation_factor(data.values, i)
            for i in range(data.shape[1])
        ]
        
        max_vif = vif['VIF'].max()
        
        if max_vif > thresh:
            drop_col = vif.loc[vif['VIF'] == max_vif, 'Feature'].values[0]
            print(f"Dropping {drop_col} with VIF {max_vif:.2f}")
            data = data.drop(columns=[drop_col])
        else:
            break
    
    return data, vif

# Separate target
y = df["TARGET"]

# Separate numeric & categorical BEFORE dummy encoding
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove("TARGET")

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

X_numeric = df[numeric_cols]
X_categorical = df[categorical_cols]

# Apply VIF only on numeric features

print("Applying VIF removal on numeric features...\n")

X_numeric_clean, final_vif = remove_high_vif(X_numeric, thresh=5.0)

print("\nFinal Numeric Features After VIF Removal:")
print(X_numeric_clean.columns)

print("\nFinal VIF Table:")
print(final_vif)

# Save VIF report
final_vif.to_csv("docs/final_vif_report.csv", index=False)

# Combine cleaned numeric + categorical

X_combined = pd.concat([X_numeric_clean, X_categorical], axis=1)

# Now apply one-hot encoding
X_final = pd.get_dummies(X_combined, drop_first=True)


# In[24]:


# Prepare VIF Interpretation

# Convert VIF to R² %
vif_data["R2_Percentage"] = (1 - (1 / vif_data["VIF"])) * 100

high_vif_features = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()

# EDA REPORT

report_part3 = f"""
PART 3 – ADVANCED FULL EDA REPORT

OBJECTIVE:
To perform comprehensive univariate, bivariate and multivariate 
analysis to identify financial, demographic and behavioral risk drivers.


UNIVARIATE ANALYSIS

Numerical Feature Distribution Statistics:
{chr(10).join(univariate_summary)}

Categorical Distribution Summary:
{chr(10).join(cat_summary)}

Key Observations:
- Financial variables show positive skewness (right-tailed distribution).
- Credit and income exhibit heavy-tailed behavior.
- Majority income type: {df['NAME_INCOME_TYPE'].mode()[0]}.
- Skewed distributions indicate potential outlier influence.


BIVARIATE ANALYSIS (Feature vs TARGET)

Mean Comparison (Default vs Non-Default):
{chr(10).join(bivariate_summary)}

Highest Risk Credit-Income Quartile:
{risk_lift.max()*100:.2f}%

Highest Risk Age Group:
{age_default.idxmax()}  |  Default Rate: {age_default.max()*100:.2f}%

Risk Insights:
- Over-leveraged customers (high credit-to-income ratio) 
  demonstrate significantly higher default probability.
- Younger age segments show relatively higher instability.
- Financial exposure strongly differentiates defaulters.


MULTIVARIATE ANALYSIS

Top 5 Positively Correlated with TARGET:
{corr_target.head().to_string()}

Top 5 Negatively Correlated with TARGET:
{corr_target.tail().to_string()}

Interpretation:
- Financial ratios and external risk sources (if available)
  dominate predictive structure.
- Limited extreme correlation observed among non-financial variables.


MULTICOLLINEARITY ANALYSIS (VIF)

Variance Inflation Factor (VIF) Table:
{vif_data.to_string(index=False)}

R² Percentage (Variance Explained by Other Predictors):
{vif_data[['Feature','R2_Percentage']].to_string(index=False)}

High VIF Features (VIF > 5):
{high_vif_features if len(high_vif_features) > 0 else "No severe multicollinearity detected."}

VIF Interpretation:
- VIF between 1–5 indicates moderate correlation.
- VIF above 5 indicates high multicollinearity.
- High R² % suggests the variable is largely explained by others.
- Financial features often show natural correlation (expected in credit datasets).

Regulatory Insight:
- For logistic regression or scorecard models, high VIF features 
  may require removal.
- For tree-based models (LightGBM), multicollinearity impact is minimal.


OVERALL CONCLUSION


- Credit-to-income ratio is a strong financial stress indicator.
- Behavioral instability and financial leverage are key risk drivers.
- Multicollinearity is manageable after VIF filtering.
- Dataset is suitable for predictive modeling and risk scoring.

"""

with open("outputs/reports/03_advanced_full_eda_report.txt",
          "w",
          encoding="utf-8") as f:
    f.write(report_part3)


# ### PART 4 – PREVIOUS APPLICATION ANALYSIS

# In[14]:


"""
Objective:
Assess behavioral credit instability.
"""


# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
prev = pd.read_csv("data/raw/previous_application.csv")


# In[16]:


# BASIC KPIs

total_prev_apps = prev.shape[0]

contract_status_dist = prev["NAME_CONTRACT_STATUS"].value_counts(normalize=True) * 100

approval_rate = contract_status_dist.get("Approved", 0)
refusal_rate = contract_status_dist.get("Refused", 0)

avg_application_amt = prev["AMT_APPLICATION"].mean()
avg_credit_amt = prev["AMT_CREDIT"].mean()
avg_annuity = prev["AMT_ANNUITY"].mean()

credit_gap = (prev["AMT_CREDIT"] - prev["AMT_APPLICATION"]).mean()

avg_payment_term = prev["CNT_PAYMENT"].mean()

insured_pct = prev["NFLAG_INSURED_ON_APPROVAL"].mean() * 100

most_common_contract = prev["NAME_CONTRACT_TYPE"].mode()[0]


# In[18]:


# CONTRACT STATUS DISTRIBUTION

plt.figure(figsize=(8,5))
prev["NAME_CONTRACT_STATUS"].value_counts().plot(kind="bar")
plt.title("Contract Status Distribution")
plt.show()
plt.savefig("outputs/figures/prev_contract_status.png", dpi=300)
plt.close()


# In[20]:


# LOAN TYPE DISTRIBUTION

plt.figure(figsize=(8,5))
prev["NAME_CONTRACT_TYPE"].value_counts().plot(kind="bar")
plt.title("Loan Type Distribution")
plt.show()
plt.savefig("outputs/figures/prev_loan_type.png", dpi=300)
plt.close()


# In[21]:


# CREDIT GAP ANALYSIS

prev["CREDIT_GAP"] = prev["AMT_CREDIT"] - prev["AMT_APPLICATION"]

plt.figure(figsize=(6,4))
sns.histplot(prev["CREDIT_GAP"], bins=50)
plt.title("Credit Gap Distribution")
plt.show()
plt.savefig("outputs/figures/prev_credit_gap.png", dpi=300)
plt.close()


# In[22]:


# PAYMENT TERM ANALYSIS

plt.figure(figsize=(6,4))
sns.histplot(prev["CNT_PAYMENT"], bins=50)
plt.title("Payment Term Distribution")
plt.show()
plt.savefig("outputs/figures/prev_payment_term.png", dpi=300)
plt.close()


# In[23]:


# WEEKDAY & HOUR ANALYSIS

plt.figure(figsize=(8,5))
prev["WEEKDAY_APPR_PROCESS_START"].value_counts().plot(kind="bar")
plt.title("Application Weekday Distribution")
plt.show()
plt.savefig("outputs/figures/prev_weekday_pattern.png", dpi=300)
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(prev["HOUR_APPR_PROCESS_START"], bins=24)
plt.title("Application Hour Distribution")
plt.show()
plt.savefig("outputs/figures/prev_hour_pattern.png", dpi=300)
plt.close()


# In[24]:


# ANOMALY CHECK – 365243 VALUES

anomaly_days = (prev["DAYS_FIRST_DRAWING"] == 365243).mean() * 100
print("Anomaly Days:",anomaly_days,"\n")

# CUSTOMER-LEVEL REFUSAL RATE

customer_refusal = prev.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"] \
    .apply(lambda x: (x == "Refused").mean()) \
    .reset_index()

customer_refusal.columns = ["SK_ID_CURR", "REFUSAL_RATE"]

high_risk_customers = (customer_refusal["REFUSAL_RATE"] > 0.5).mean() * 100
print("\nHigh Risk Customers: ",high_risk_customers)

# REPORT GENERATION

report_prev = f"""
PREVIOUS APPLICATION ANALYSIS REPORT

Total Previous Applications: {total_prev_apps}

Contract Status:
- Approval Rate: {approval_rate:.2f}%
- Refusal Rate: {refusal_rate:.2f}%

Financial Metrics:
- Avg Application Amount: {avg_application_amt:,.2f}
- Avg Credit Amount: {avg_credit_amt:,.2f}
- Avg Credit Gap: {credit_gap:,.2f}
- Avg Annuity: {avg_annuity:,.2f}
- Avg Payment Term: {avg_payment_term:.2f} months

Insurance Behavior:
- % Loans Insured: {insured_pct:.2f}%

Loan Type:
- Most Common Loan Type: {most_common_contract}

Data Anomaly:
- % Records with 365243 anomaly in DAYS_FIRST_DRAWING: {anomaly_days:.2f}%

Behavioral Risk:
- % Customers with >50% refusal rate: {high_risk_customers:.2f}%

Key Insights:
- High refusal rate indicates behavioral instability.
- Credit gap suggests approval adjustments by lender.
- Long payment terms may correlate with higher exposure.
- 365243 values indicate placeholder or missing date encoding.
"""

with open("outputs/reports/previous_application_analysis_report.txt",
          "w",
          encoding="utf-8") as f:
    f.write(report_prev)

print("Previous Application Analysis Completed Successfully.")


# ### PART 5 – LIGHTGBM MODEL

# In[25]:


"""
OBJECTIVE:
Optimize LightGBM hyperparameters using Optuna
after VIF cleaning and SMOTE balancing.

DESCRIPTION:
- Apply SMOTE only on training data
- Tune hyperparameters using AUC
- Retrain best model
- Evaluate using AUC & KS
- Save model for deployment
"""


# In[35]:


import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")


# In[37]:


X_combined = pd.concat([X_numeric_clean, X_categorical], axis=1)
X_final = pd.get_dummies(X_combined, drop_first=True)
y = df["TARGET"]

def clean_feature_names(columns):
    return [re.sub(r"[^A-Za-z0-9_]", "_", col) for col in columns]

X_final.columns = clean_feature_names(X_final.columns)


# In[38]:


# As the target column is highly imbalanced we will use smote to balance it and apply optuna tuning with 5 fold CV

def objective(trial):

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 120),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
        "random_state": 42
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in skf.split(X_final, y):

        X_train_fold = X_final.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X_final.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Apply SMOTE ONLY on training fold
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(
            X_train_fold, y_train_fold
        )

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train_smote,
            y_train_smote,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)
            ]
        )

        preds = model.predict_proba(X_val_fold)[:, 1]
        fold_auc = roc_auc_score(y_val_fold, preds)
        auc_scores.append(fold_auc)

        # Pruning
        trial.report(np.mean(auc_scores), len(auc_scores))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(auc_scores)


# In[39]:


# Run Optuna Study

optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner()
)

study.optimize(objective, n_trials=25)

print("Best CV AUC:", study.best_value)
print("Best Parameters:", study.best_params)


# In[40]:


# Train Final Model on Full Dataset

best_params = study.best_params
best_params.update({
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1
})

final_model = lgb.LGBMClassifier(**best_params)

# Apply SMOTE on full dataset
smote = SMOTE(random_state=42)
X_smote_full, y_smote_full = smote.fit_resample(X_final, y)

final_model.fit(X_smote_full, y_smote_full)


# In[41]:


# Feature Importance

importance = pd.DataFrame({
    "Feature": X_final.columns,
    "Importance": final_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance.to_csv("docs/feature_importance_optuna_cv.csv", index=False)


# In[44]:


from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

# Predict probabilities
y_scores = final_model.predict_proba(X_val)[:, 1]


# In[45]:


precision, recall, thresholds = precision_recall_curve(y_val, y_scores)

ap_score = average_precision_score(y_val, y_scores)

print("Average Precision (AP):", ap_score)


# In[46]:


plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"AP = {ap_score:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.savefig("outputs/figures/precision_recall_curve.png", dpi=300)
plt.show()


# In[47]:


from sklearn.metrics import f1_score

f1_scores = []

for t in thresholds:
    preds = (y_scores >= t).astype(int)
    f1_scores.append(f1_score(y_val, preds))

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("Best Threshold (F1):", best_threshold)
print("Best F1 Score:", f1_scores[best_index])


# In[48]:


from sklearn.metrics import classification_report

final_preds = (y_scores >= best_threshold).astype(int)

print(classification_report(y_val, final_preds))


# In[49]:


target_recall = 0.80
idx = np.argmin(np.abs(recall - target_recall))
print("Threshold at 80% Recall:", thresholds[idx])
print("Precision at 80% Recall:", precision[idx])


# In[50]:


# Save Model

joblib.dump(final_model, "model/credit_risk_model_optuna_cv.pkl")


# In[64]:


# Prepare Additional Metrics for Report

from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_curve
)

# Validation predictions
y_scores = final_model.predict_proba(X_final)[:, 1]
print("Min probability:", y_scores.min())
print("Max probability:", y_scores.max())


# In[65]:


# Precision-Recall

precision, recall, thresholds = precision_recall_curve(y, y_scores)
ap_score = average_precision_score(y, y_scores)

print("Average Precision (PR AUC):", ap_score)


# In[66]:


# F1 Threshold Optimization

f1_scores = 2 * (precision[:-1] * recall[:-1]) / (
    precision[:-1] + recall[:-1] + 1e-10
)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print("Best Threshold:", best_threshold)
print("Best F1 Score:", best_f1)


# In[67]:


# KS Statistic

fpr, tpr, _ = roc_curve(y, y_scores)
ks_stat = np.max(tpr - fpr)

print("KS Statistic:", ks_stat)


# In[68]:


# Classification report at best threshold

final_preds = (y_scores >= best_threshold).astype(int)

class_report = classification_report(y, final_preds)

print("\nClassification Report:\n")
print(class_report)


# In[69]:


# FINAL REPORT

report = f"""
LIGHTGBM + SMOTE + OPTUNA (5-FOLD CV) – FINAL MODEL REPORT

MODEL OBJECTIVE

Develop a robust credit risk classification model capable of 
detecting high-risk (default) applicants using advanced 
feature engineering, imbalance handling and hyperparameter tuning.



MODEL PERFORMANCE SUMMARY


Best Cross-Validated AUC (5-Fold): {study.best_value:.4f}
KS Statistic: {ks_stat:.4f}
Average Precision (PR AUC): {ap_score:.4f}

Optimal Threshold (Max F1): {best_threshold:.4f}
Best F1 Score: {best_f1:.4f}

Classification Report (at optimal threshold):
{class_report}



BEST HYPERPARAMETERS (Optuna)

{study.best_params}



TOP 5 IMPORTANT FEATURES

{importance.head().to_string(index=False)}

Interpretation:
- Features with highest importance contribute most to 
  distinguishing defaulters from non-defaulters.
- Financial exposure and behavioral instability variables 
  typically dominate importance ranking.


IMBALANCE HANDLING

- SMOTE applied inside each cross-validation fold
- Prevented data leakage
- Improved minority class recall



PIPELINE STRUCTURE

1. Data Cleaning & Missing Value Treatment
2. Feature Engineering & Ratio Creation
3. VIF-Based Multicollinearity Reduction
4. One-Hot Encoding with Feature Name Sanitization
5. SMOTE Balancing (Training Fold Only)
6. Stratified 5-Fold Cross Validation
7. Optuna Hyperparameter Optimization (with pruning)
8. Final Model Training on Balanced Dataset
9. Deployment Model Serialization



BUSINESS INTERPRETATION

- High Recall improves identification of risky borrowers.
- Precision-Recall analysis helps define approval cutoff.
- KS Statistic indicates separation power between risk groups.
- Model suitable for risk scoring and decision support systems.



DEPLOYMENT

Saved Model Path:
data/processed/credit_risk_model_optuna_cv.pkl

Model is ready for inference on unseen application data.


"""

with open("outputs/reports/07_optuna_cv_model_report.txt",
          "w",
          encoding="utf-8") as f:
    f.write(report)

print("FINAL MODEL REPORT GENERATED SUCCESSFULLY.")

