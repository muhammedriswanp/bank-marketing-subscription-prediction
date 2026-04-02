import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# 1. Set the experiment name
mlflow.set_experiment("bank-marketing-subscription")

def train_and_log_model(model_name, pipeline, X_train, y_train, X_test, y_test, params):
    # Start an MLflow run
    with mlflow.start_run(run_name=model_name):
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        acc = accuracy_score(y_test, y_pred)
        
        # 2. Log Parameters (Hyperparameters)
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        
        # 3. Log Metrics
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("roc_auc", round(auc, 4))
        mlflow.log_metric("accuracy", round(acc, 4))
        
        # 4. Log the Model (The entire pipeline!)
        mlflow.sklearn.log_model(pipeline, "model_pipeline")
        
        print(f"Logged {model_name} to MLflow: F1={f1:.4f}")