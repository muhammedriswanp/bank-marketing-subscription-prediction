# Bank Marketing Subscription Prediction

Predict whether a client will subscribe to a **term deposit** based on direct marketing campaign data from a Portuguese bank.

## Problem Statement

Binary classification on the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) — 41,188 records, 20 features, target variable `y` (`yes` / `no`).

## Dataset

`data/bank-additional-full.csv` — Phone-call campaign data (May 2008 – Nov 2010).  
Features include client demographics, last contact info, campaign history, and socio-economic indicators.

## Approach

- **EDA** — Distribution analysis, class imbalance check, feature correlations
- **Preprocessing** — Encoding, scaling, train/test split
- **Modeling** — Classification models with evaluation metrics (accuracy, F1, ROC-AUC)

## Project Structure

```
├── data/                  # Dataset
├── notebook/
│   └── eda.ipynb          # Exploratory Data Analysis
└── src/
    ├── preprocessing.py   # Data cleaning & feature engineering
    ├── model.py           # Model training & evaluation
    └── utils.py           # Helper functions
```

## Run Locally

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook notebook/eda.ipynb
```