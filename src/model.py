import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import build_preprocessor
from src.utils import save_model

# ── Load data ──────────────────────────────────────────────────────────────────
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

preprocessor = build_preprocessor()

# ── Define models ──────────────────────────────────────────────────────────────
models = {
    'Baseline — Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        class_weight='balanced', random_state=42
    ),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=100, random_state=42
    )
}

# ── Train, evaluate, save ──────────────────────────────────────────────────────
results = []

for name, clf in models.items():
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    # cross-validation on train set
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring='f1', n_jobs=-1
    )
    print(f"  CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # fit on full train set
    pipeline.fit(X_train, y_train)
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_pred_prob)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    results.append({
        'Model'    : name,
        'CV F1'    : round(cv_scores.mean(), 4),
        'Accuracy' : round(acc, 4),
        'Precision': round(precision, 4),
        'Recall'   : round(recall, 4),
        'F1'       : round(f1, 4),
        'ROC-AUC'  : round(roc_auc, 4)
    })

    # save each pipeline
    filename = name.lower().replace(' ', '_').replace('—', '').replace('  ', '_') + '.pkl'
    save_model(pipeline, filename)

# ── Comparison table ───────────────────────────────────────────────────────────
print("\n\n" + "="*55)
print("  MODEL COMPARISON")
print("="*55)
results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
print(results_df.to_string(index=False))

os.makedirs('outputs', exist_ok=True)
results_df.to_csv('outputs/model_comparison.csv', index=False)
print("\nSaved → outputs/model_comparison.csv")