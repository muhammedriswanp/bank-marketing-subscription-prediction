import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score, accuracy_score
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_model

# ── Setup MLflow ───────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-subscription")

# ── Load test data ─────────────────────────────────────────────────────────────
X_test  = pd.read_csv('data/X_test.csv')
y_test  = pd.read_csv('data/y_test.csv').squeeze()

# ── Models to log ──────────────────────────────────────────────────────────────
experiments = [
    {
        'name'      : 'Baseline — Logistic Regression',
        'file'      : 'baseline__logistic_regression.pkl',
        'params'    : {
            'model_type'    : 'LogisticRegression',
            'max_iter'      : 1000,
            'class_weight'  : 'balanced'
        }
    },
    {
        'name'      : 'Decision Tree',
        'file'      : 'decision_tree.pkl',
        'params'    : {
            'model_type'    : 'DecisionTreeClassifier',
            'class_weight'  : 'balanced'
        }
    },
    {
        'name'      : 'KNN',
        'file'      : 'knn.pkl',
        'params'    : {
            'model_type'    : 'KNeighborsClassifier',
            'n_neighbors'   : 5
        }
    },
    {
        'name'      : 'Random Forest',
        'file'      : 'random_forest.pkl',
        'params'    : {
            'model_type'    : 'RandomForestClassifier',
            'n_estimators'  : 100,
            'class_weight'  : 'balanced'
        }
    },
    {
        'name'      : 'Gradient Boosting',
        'file'      : 'gradient_boosting.pkl',
        'params'    : {
            'model_type'    : 'GradientBoostingClassifier',
            'n_estimators'  : 100,
            'learning_rate' : 0.1
        }
    },
    {
        'name'      : 'AdaBoost',
        'file'      : 'adaboost.pkl',
        'params'    : {
            'model_type'    : 'AdaBoostClassifier',
            'n_estimators'  : 100,
            'learning_rate' : 1.0
        }
    },
    {
        'name'      : 'Random Forest (Tuned)',
        'file'      : 'random_forest_tuned.pkl',
        'params'    : {
            'model_type'        : 'RandomForestClassifier',
            'n_estimators'      : 200,
            'max_depth'         : 20,
            'min_samples_split' : 10,
            'class_weight'      : 'balanced',
            'tuned'             : True
        }
    },
    {
        'name'      : 'Gradient Boosting (Tuned)',
        'file'      : 'gradient_boosting_tuned.pkl',
        'params'    : {
            'model_type'    : 'GradientBoostingClassifier',
            'n_estimators'  : 100,
            'learning_rate' : 0.2,
            'max_depth'     : 5,
            'tuned'         : True
        }
    }
]

# ── Log each model ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Logging experiments to MLflow")
print("="*60)

for exp in experiments:
    with mlflow.start_run(run_name=exp['name']):

        # load pipeline
        pipeline    = load_model(exp['file'])
        y_pred      = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

        # calculate metrics
        metrics = {
            'accuracy'  : round(accuracy_score(y_test, y_pred), 4),
            'precision' : round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall'    : round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1_score'  : round(f1_score(y_test, y_pred), 4),
            'roc_auc'   : round(roc_auc_score(y_test, y_pred_prob), 4)
        }

        # log params and metrics
        mlflow.log_params(exp['params'])
        mlflow.log_metrics(metrics)

        # log pipeline as model
        mlflow.sklearn.log_model(pipeline, artifact_path="pipeline")

        print(f"\n  ✅ {exp['name']}")
        print(f"     F1: {metrics['f1_score']} | "
              f"ROC-AUC: {metrics['roc_auc']} | "
              f"Recall: {metrics['recall']}")

print("\n" + "="*60)
print("  All experiments logged!")
print("  Run: mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("="*60)