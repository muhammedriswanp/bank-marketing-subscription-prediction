import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


# ── Column definitions ─────────────────────────────────────────────────────────

COLS_TO_DROP = ['duration', 'default', 'pdays']

NUMERICAL_FEATURES = [
    'age', 'campaign', 'previous',
    'emp.var.rate', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m', 'nr.employed'
]

ORDINAL_FEATURE = ['education']
ORDINAL_CATEGORIES = [[
    'illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
    'high.school', 'professional.course', 'university.degree', 'unknown'
]]

OHE_FEATURES = [
    'job', 'marital', 'contact',
    'month', 'day_of_week', 'poutcome',
    'housing', 'loan'
]


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # previous_contact: was client contacted before this campaign?
    df['previous_contact'] = (df['pdays'] != 999).astype(int)

    # replace 'unknown' strings with NaN for imputation
    df.replace('unknown', np.nan, inplace=True)

    # drop leakage + redundant columns
    df.drop(columns=COLS_TO_DROP, inplace=True)

    return df


# ── Preprocessing pipeline ─────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    ohe_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # add previous_contact to numerical after feature engineering
    numerical_cols = NUMERICAL_FEATURES + ['previous_contact']

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('ord', ordinal_pipeline, ORDINAL_FEATURE),
        ('ohe', ohe_pipeline, OHE_FEATURES)
    ])

    return preprocessor


# ── Target encoding ────────────────────────────────────────────────────────────

def encode_target(series: pd.Series) -> pd.Series:
    return series.map({'yes': 1, 'no': 0})