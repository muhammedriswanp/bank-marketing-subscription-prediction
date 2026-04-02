import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_model

# ── Load data ──────────────────────────────────────────────────────────────────
X_test  = pd.read_csv('data/X_test.csv')
y_test  = pd.read_csv('data/y_test.csv').squeeze()

os.makedirs('outputs/plots', exist_ok=True)

# ── Load all models ────────────────────────────────────────────────────────────
models = {
    'Logistic Regression (Baseline)' : load_model('baseline__logistic_regression.pkl'),
    'Decision Tree'                   : load_model('decision_tree.pkl'),
    'KNN'                             : load_model('knn.pkl'),
    'Random Forest'                   : load_model('random_forest.pkl'),
    'Gradient Boosting'               : load_model('gradient_boosting.pkl'),
    'AdaBoost'                        : load_model('adaboost.pkl'),
    'Random Forest (Tuned)'           : load_model('random_forest_tuned.pkl'),
    'Gradient Boosting (Tuned)'       : load_model('gradient_boosting_tuned.pkl'),
}

# ── Full evaluation ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  FULL MODEL EVALUATION")
print("="*60)


results = []
for name, pipeline in models.items():
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall'   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1'       : round(f1_score(y_test, y_pred), 4),
        'ROC-AUC'  : round(roc_auc_score(y_test, y_pred_prob), 4)
    })


results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv('outputs/final_evaluation.csv', index=False)
print("\nSaved → outputs/final_evaluation.csv")

# ── Confusion Matrix — Best Model ──────────────────────────────────────────────
print("\n" + "="*60)
print("  CONFUSION MATRIX — Random Forest (Tuned)")
print("="*60)

best_model = models['Random Forest (Tuned)']
y_pred     = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Random Forest (Tuned)')
plt.tight_layout()
plt.savefig('outputs/plots/confusion_matrix.png', dpi=150)
plt.close()
print("Saved → outputs/plots/confusion_matrix.png")

# ── ROC Curve — All Models ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ROC CURVES — All Models")
print("="*60)

fig, ax = plt.subplots(figsize=(9, 6))
for name, pipeline in models.items():
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(
        y_test, y_pred_prob, name=name, ax=ax
    )
ax.set_title('ROC Curves — All Models')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.tight_layout()
plt.savefig('outputs/plots/roc_curves.png', dpi=150)
plt.close()
print("Saved → outputs/plots/roc_curves.png")

# ── Feature Importance ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  FEATURE IMPORTANCE — Random Forest (Tuned)")
print("="*60)

rf_pipeline    = models['Random Forest (Tuned)']
rf_classifier  = rf_pipeline.named_steps['classifier']
preprocessor   = rf_pipeline.named_steps['preprocessor']

# get feature names after preprocessing
num_features = preprocessor.transformers_[0][2]
ord_features = preprocessor.transformers_[1][2]
ohe_features = preprocessor.transformers_[2][1].named_steps['encoder']\
                            .get_feature_names_out(
                                preprocessor.transformers_[2][2]
                            ).tolist()

all_features = list(num_features) + list(ord_features) + ohe_features

importances = rf_classifier.feature_importances_
feat_df = pd.DataFrame({
    'Feature'   : all_features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feat_df.head(10).to_string(index=False))

# plot top 15
fig, ax = plt.subplots(figsize=(9, 6))
top15 = feat_df.head(15)
sns.barplot(data=top15, x='Importance', y='Feature', 
            hue='Feature', palette='viridis', legend=False, ax=ax)
ax.set_title('Top 15 Feature Importances — Random Forest (Tuned)')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/plots/feature_importance.png', dpi=150)
plt.close()
print("\nSaved → outputs/plots/feature_importance.png")

feat_df.to_csv('outputs/feature_importance.csv', index=False)
print("Saved → outputs/feature_importance.csv")

# ── Overfitting Check ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  OVERFITTING CHECK — Random Forest (Tuned)")
print("="*60)

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()

train_f1 = f1_score(y_train, best_model.predict(X_train))
test_f1  = f1_score(y_test,  best_model.predict(X_test))

print(f"  Train F1 : {train_f1:.4f}")
print(f"  Test F1  : {test_f1:.4f}")
print(f"  Gap      : {train_f1 - test_f1:.4f}")

if train_f1 - test_f1 > 0.1:
    print("  ⚠️  Overfitting detected — large gap between train and test")
else:
    print("  ✅ No significant overfitting")