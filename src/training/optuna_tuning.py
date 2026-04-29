import optuna
import mlflow
import logging
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from train import load_and_prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-risk-optuna-tuning")

# Load data globally for Optuna to use across trials
X, y = load_and_prepare_data()
imbalance_ratio = (y == 0).sum() / (y == 1).sum()


def objective(trial):
    with mlflow.start_run(nested=True):
        # 1. Define hyperparameter search space
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "scale_pos_weight": imbalance_ratio,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }

        mlflow.log_params(params)

        tscv = TimeSeriesSplit(n_splits=3)
        cv_auc = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            cv_auc.append(auc)

        mean_auc = np.mean(cv_auc)
        mlflow.log_metric("cv_mean_roc_auc", mean_auc)

        return mean_auc


def main():
    logging.info("Starting Optuna optimization for LightGBM...")

    # Create an MLflow run to group all the nested Optuna trials
    with mlflow.start_run(run_name="LightGBM_Optuna_Study"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)  # Keep trials low for local execution

        logging.info(f"Best trial ROC-AUC: {study.best_value:.4f}")
        logging.info("Best parameters:")
        for key, value in study.best_params.items():
            logging.info(f"  {key}: {value}")
            mlflow.log_param(f"best_{key}", value)

        mlflow.log_metric("best_cv_roc_auc", study.best_value)


if __name__ == "__main__":
    main()
