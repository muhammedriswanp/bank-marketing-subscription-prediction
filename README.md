# Bank Marketing Subscription Prediction 🏦
Predicts whether a bank client will subscribe to a term deposit.
---

## Problem Statement
Given client demographics, campaign details, and economic indicators, predict if a client will subscribe to a term deposit (`yes` / `no`) using direct marketing campaign data from a Portuguese bank.

## Dataset
- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **File:** `bank-additional-full.csv`
- **Size:** 41,188 rows · 20 features
- **Features:** Numerical, categorical, economic indicators
- **Challenge:** Class imbalance — 88.7% `no` vs 11.3% `yes`

## Approach
1. **EDA** — explored distributions, class imbalance, unknown values, outliers, correlations
2. **Preprocessing** — dropped leakage/low-variance columns, engineered `previous_contact` from `pdays`, mode imputation, OrdinalEncoder for `education`, OneHotEncoder for categoricals, StandardScaler for numericals
3. **Modeling** — 6 models trained inside sklearn `Pipeline` + `ColumnTransformer`
4. **Tuning** — `GridSearchCV` with 5-fold CV on best 2 models, scored on `f1`
5. **Deployment** — Flask REST API + Streamlit dashboard + Docker + GitHub Actions CI

## Key Preprocessing Decisions
```
duration         → dropped    (data leakage — unknown before call ends)
default          → dropped    (only 3 'yes' values out of 32,950)
pdays            → dropped    (replaced by engineered feature)
previous_contact → engineered (1 if client was contacted before, else 0)
unknown strings  → treated as NaN, imputed with mode
class imbalance  → class_weight='balanced' in all models
metrics          → f1-score + roc-auc (accuracy is misleading here)
```

## Models Used

| Model | Type |
|-------|------|
| Logistic Regression | Baseline — Linear |
| Decision Tree | Tree-based |
| KNN | Distance-based |
| Random Forest | Ensemble — Bagging |
| Gradient Boosting | Ensemble — Boosting |
| AdaBoost | Ensemble — Boosting |
| Random Forest (Tuned) ⭐ | GridSearchCV Best |
| Gradient Boosting (Tuned) | GridSearchCV |

## Results
**Best Model: Random Forest (Tuned)**
- Primary metrics: **F1-score** + **ROC-AUC**
- Top features: `euribor3m`, `age`, `nr.employed`, `emp.var.rate`, `campaign`
- Full scores in `outputs/final_evaluation.csv`

## Project Structure
```
bank-marketing-subscription-prediction/
├── data/
│   ├── bank-additional-full.csv
│   ├── train.csv
│   └── test.csv
├── notebook/
│   └── eda.ipynb
├── src/
│   ├── preprocessing.py       # ColumnTransformer pipeline
│   ├── model.py               # Model training + comparison
│   ├── evaluate.py            # Metrics, plots, feature importance
│   ├── tune.py                # GridSearchCV tuning
│   └── utils.py               # save/load model helpers
├── models/
│   └── random_forest_tuned.pkl
├── outputs/
│   ├── final_evaluation.csv
│   ├── feature_importance.csv
│   └── plots/
├── app.py                     # Streamlit dashboard
├── flask_api.py               # REST API
├── Dockerfile.flask           # Docker container
└── requirements.txt
```

## How to Run Locally
```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. split data
python src/train_test_split.py

# 3. train models
python src/model.py

# 4. tune
python src/tune.py

# 5. evaluate
python src/evaluate.py
```

## Streamlit App
```bash
streamlit run app.py
```

## Flask API
```bash
python flask_api.py
python test_flask_api.py   # run tests
```

## Docker
```bash
docker build -f Dockerfile.flask -t bank-marketing-api .
docker run -v $(pwd)/models:/app/models -p 5000:5000 bank-marketing-api
```

## MLflow Tracking
```bash
python log_experiment.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Stack
```
sklearn · pandas · flask · streamlit · mlflow · dvc · docker · github actions
```
