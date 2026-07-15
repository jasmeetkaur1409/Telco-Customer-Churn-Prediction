# Telco Customer Churn Prediction

Predicting which telecom customers are likely to churn, and identifying who a retention team
should prioritize first — using the IBM Telco Customer Churn dataset (7,043 customers, 21 columns).

## Business questions this project answers
1. How big is the churn problem — what % of customers are churning overall?
2. Which customer segments (contract type, tenure, internet service, payment method, etc.) churn the most?
3. Can we predict, for an individual customer, whether they're likely to churn — and how reliable is that prediction?
4. What factors does the model rely on most when it predicts churn?
5. Which specific customer profile should a retention team prioritize first?

Full answers, backed by the actual notebook output, are in the **"Business Questions – Answered"**
section at the end of the notebook.

## Key results
- **Churn rate:** 26.6% of customers in the dataset churned
- **Best model:** Tuned XGBoost — 83.4% ROC-AUC, 70% recall (0.5 threshold)
- **Threshold-adjusted:** recall raised to 80% by lowering the decision threshold to 0.391,
  trading off some precision — see the notebook's Threshold Tuning section for the reasoning
- **Top churn drivers:** contract type, tenure, internet service type, payment method, number of add-on services
- **Highest-risk segment:** new customers (<12 months tenure), month-to-month contract,
  paying by electronic check, with few or no add-on services

## Approach
- Data cleaning: fixed inconsistent categorical values, handled missing `TotalCharges`, removed
  zero-tenure rows, de-duplication check
- EDA: churn rate by contract type, tenure, monthly charges, internet service, payment method
- Feature engineering: tenure buckets, `num_services` (add-on count), `charge_per_tenure`
- Preprocessing: `ColumnTransformer` (scaling + one-hot encoding) inside a pipeline to avoid
  train/test leakage
- Class imbalance handled with **SMOTE**, applied only inside the training pipeline
- Compared Logistic Regression, Random Forest, XGBoost, and SVM before tuning
- Hyperparameter tuning via `RandomizedSearchCV` (5-fold stratified CV) on XGBoost, plus a tuned
  Logistic Regression for direct comparison
- Threshold tuning to prioritize recall for the retention use case
- Feature importance to explain what the model relies on most

## Tools
Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn (SMOTE), Matplotlib, Seaborn, Jupyter Notebook

## Dataset
[Telco Customer Churn (IBM sample dataset)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — widely used on Kaggle.

## Author
Jasmeet — [GitHub](https://github.com/jasmeetkaur1409)
