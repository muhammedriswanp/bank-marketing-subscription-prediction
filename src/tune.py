import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import build_preprocessor
from src.utils import save_model

# ── Load data ──────────────────────────────────────────────────────────────────
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

# ── Random Forest Tuning ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Tuning: Random Forest")
print("="*55)

rf_pipeline = Pipeline([
    ('preprocessor', build_preprocessor()),
    ('classifier', RandomForestClassifier(
        class_weight='balanced', random_state=42
    ))
])

rf_param_grid = {
    'classifier__n_estimators'   : [100, 200],
    'classifier__max_depth'      : [5, 10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

print(f"\n  Best Params : {rf_grid.best_params_}")
print(f"  Best CV F1  : {rf_grid.best_score_:.4f}")

rf_best = rf_grid.best_estimator_
y_pred      = rf_best.predict(X_test)
y_pred_prob = rf_best.predict_proba(X_test)[:, 1]

print(f"\n  Test F1     : {f1_score(y_test, y_pred):.4f}")
print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

save_model(rf_best, 'random_forest_tuned.pkl')


# ── Gradient Boosting Tuning ───────────────────────────────────────────────────
print("\n" + "="*55)
print("  Tuning: Gradient Boosting")
print("="*55)

gb_pipeline = Pipeline([
    ('preprocessor', build_preprocessor()),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

gb_param_grid = {
    'classifier__n_estimators' : [100, 200],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__max_depth'    : [3, 5, 7],
}

gb_grid = GridSearchCV(
    gb_pipeline,
    gb_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

gb_grid.fit(X_train, y_train)

print(f"\n  Best Params : {gb_grid.best_params_}")
print(f"  Best CV F1  : {gb_grid.best_score_:.4f}")

gb_best = gb_grid.best_estimator_
y_pred      = gb_best.predict(X_test)
y_pred_prob = gb_best.predict_proba(X_test)[:, 1]

print(f"\n  Test F1     : {f1_score(y_test, y_pred):.4f}")
print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

save_model(gb_best, 'gradient_boosting_tuned.pkl')


# ── Final Comparison ───────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  TUNED MODEL COMPARISON")
print("="*55)

baseline = pd.read_csv('outputs/model_comparison.csv')

tuned_results = pd.DataFrame([
    {
        'Model'    : 'Random Forest (Tuned)',
        'CV F1'    : round(rf_grid.best_score_, 4),
        'F1'       : round(f1_score(y_test, rf_best.predict(X_test)), 4),
        'ROC-AUC'  : round(roc_auc_score(y_test, rf_best.predict_proba(X_test)[:, 1]), 4)
    },
    {
        'Model'    : 'Gradient Boosting (Tuned)',
        'CV F1'    : round(gb_grid.best_score_, 4),
        'F1'       : round(f1_score(y_test, gb_best.predict(X_test)), 4),
        'ROC-AUC'  : round(roc_auc_score(y_test, gb_best.predict_proba(X_test)[:, 1]), 4)
    }
])

print(tuned_results.to_string(index=False))

os.makedirs('outputs', exist_ok=True)
tuned_results.to_csv('outputs/tuned_model_comparison.csv', index=False)
print("\nSaved → outputs/tuned_model_comparison.csv")