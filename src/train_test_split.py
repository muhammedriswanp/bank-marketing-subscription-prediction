import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import engineer_features, encode_target

df = pd.read_csv('data/bank-additional-full.csv', sep=';')

# engineer features before split
df = engineer_features(df)

# encode target
X = df.drop(columns=['y'])
y = encode_target(df['y'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

os.makedirs('data', exist_ok=True)
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Target distribution (train):\n{y_train.value_counts(normalize=True).round(3)}")