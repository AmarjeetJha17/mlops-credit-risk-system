import pandas as pd
import numpy as np
import os
import sys
import joblib
import logging
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

load_dotenv()

# Add src/ to path so pickled pipeline can resolve the 'features' module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_DATA_PATH = "data/raw/application_train.csv"
PIPELINE_PATH = "models/preprocessing_pipeline.joblib"
TOP_FEATURES_PATH = "models/top_features.joblib"
ARTIFACTS_DIR = "reports/figures"

# 1. Initialize MLflow with Azure ML backend
tracking_uri = os.getenv("AZURE_ML_MLFLOW_URI")
if not tracking_uri:
    raise ValueError("AZURE_ML_MLFLOW_URI environment variable not set.")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("credit-risk-baselines")


def load_and_prepare_data():
    """Loads raw data, sorts chronologically, and applies the saved pipeline."""
    logging.info("Loading and sorting data chronologically by SK_ID_CURR...")
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.sort_values("SK_ID_CURR").reset_index(drop=True)

    y = df["TARGET"]
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])

    logging.info("Loading pipeline and top features...")
    pipeline = joblib.load(PIPELINE_PATH)
    top_features = joblib.load(TOP_FEATURES_PATH)

    # Transform data
    X_transformed_array = pipeline.transform(X)
    raw_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Sanitize feature names (the fix we applied earlier)
    import re

    clean_feature_names = [
        re.sub(r"[^A-Za-z0-9_]+", "_", name) for name in raw_feature_names
    ]

    X_transformed = pd.DataFrame(X_transformed_array, columns=clean_feature_names)

    # Filter down to top features to speed up baseline training
    X_final = X_transformed[top_features]

    return X_final, y


def generate_and_log_plots(model, X_val, y_val, model_name):
    """Generates confusion matrix and SHAP plots, logging them to MLflow."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Confusion Matrix
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    cm_path = f"{ARTIFACTS_DIR}/{model_name}_cm.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # 2. SHAP Plot (Only for tree-based models to save time)
    if model_name in ["LightGBM", "XGBoost", "RandomForest"]:
        # Use a background sample to speed up SHAP calculation
        X_sample = X_val.sample(n=min(1000, len(X_val)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification outputs
        shap_target = shap_values[1] if isinstance(shap_values, list) else shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_target, X_sample, show=False)
        plt.title(f"{model_name} SHAP Summary")
        shap_path = f"{ARTIFACTS_DIR}/{model_name}_shap.png"
        plt.savefig(shap_path, bbox_inches="tight")
        mlflow.log_artifact(shap_path)
        plt.close()


def main():
    X, y = load_and_prepare_data()
    imbalance_ratio = (y == 0).sum() / (y == 1).sum()

    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            scale_pos_weight=imbalance_ratio,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=imbalance_ratio,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }

    tscv = TimeSeriesSplit(n_splits=3)  # Simulates time-based out-of-sample validation

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_baseline"):
            logging.info(f"Training {model_name}...")

            mlflow.log_param("model_type", model_name)
            mlflow.log_params(model.get_params())

            # Cross-validation arrays
            auc_scores, f1_scores, log_losses = [], [], []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = model.predict(X_val)

                auc_scores.append(roc_auc_score(y_val, y_pred_proba))
                f1_scores.append(f1_score(y_val, y_pred))
                log_losses.append(log_loss(y_val, y_pred_proba))

            # Log CV Metrics
            mlflow.log_metric("cv_mean_roc_auc", np.mean(auc_scores))
            mlflow.log_metric("cv_mean_f1", np.mean(f1_scores))
            mlflow.log_metric("cv_mean_logloss", np.mean(log_losses))

            logging.info(f"{model_name} CV ROC-AUC: {np.mean(auc_scores):.4f}")

            # Train final model on full dataset for artifact logging
            model.fit(X, y)
            generate_and_log_plots(model, X_val, y_val, model_name)

            # Log the model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
