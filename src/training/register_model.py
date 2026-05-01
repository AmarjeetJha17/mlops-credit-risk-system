import mlflow
import os
from dotenv import load_dotenv
import logging
from mlflow.tracking import MlflowClient
from datetime import datetime
import sys

load_dotenv()

# Add src/ to path so pickled pipeline can resolve the 'features' module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set tracking URI to our local SQLite DB
tracking_uri = os.getenv("AZURE_ML_MLFLOW_URI")
if not tracking_uri:
    raise ValueError("AZURE_ML_MLFLOW_URI environment variable not set.")
mlflow.set_tracking_uri(tracking_uri)
MODEL_NAME = "CreditRiskModel"
AUC_THRESHOLD = 0.7500


def get_best_run(client: MlflowClient, experiment_name: str):
    """Finds the run with the highest cross-validated ROC-AUC."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=50,
    )

    # Filter to only FINISHED runs that have the AUC metric
    finished_runs = [
        r
        for r in runs
        if r.info.status == "FINISHED"
        and r.data.metrics.get("cv_mean_roc_auc") is not None
    ]

    if not finished_runs:
        return None

    # Sort client-side by AUC descending
    finished_runs.sort(key=lambda r: r.data.metrics["cv_mean_roc_auc"], reverse=True)
    return finished_runs[0]


def validate_and_promote():
    client = MlflowClient()

    # 1. Find the best model from our baseline or tuning experiments
    logging.info("Searching for the best model run...")
    best_run = get_best_run(
        client, "credit-risk-baselines"
    )  # Or 'credit-risk-optuna-tuning'

    if not best_run:
        logging.error("No runs found.")
        return

    best_auc = best_run.data.metrics.get("cv_mean_roc_auc", 0)
    run_id = best_run.info.run_id
    model_type = best_run.data.params.get("model_type", "Unknown")

    logging.info(f"Best Run ID: {run_id} ({model_type}) with AUC: {best_auc:.4f}")

    # 2. Validation Step: Does it pass our business threshold?
    if best_auc < AUC_THRESHOLD:
        logging.warning(
            f"Model AUC {best_auc:.4f} is below threshold {AUC_THRESHOLD}. Aborting promotion."
        )
        return

    logging.info("Model passed validation. Proceeding to registration.")

    # 3. Register the Model
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version_num = model_version.version
    logging.info(f"Registered {MODEL_NAME} as Version {version_num}.")

    # 4. Add Rich Metadata/Tags
    client.set_registered_model_tag(MODEL_NAME, "dataset", "HomeCredit_v1")
    client.set_model_version_tag(
        MODEL_NAME, version_num, "training_date", datetime.now().strftime("%Y-%m-%d")
    )
    client.set_model_version_tag(MODEL_NAME, version_num, "roc_auc", str(best_auc))
    client.set_model_version_tag(MODEL_NAME, version_num, "model_type", model_type)

    # Optional: Log features used if tracked in params
    client.set_model_version_tag(MODEL_NAME, version_num, "features", "top_50_shap")

    # 5. Manage Lifecycle: Staging -> Production Workflow
    logging.info("Managing Staging/Production lifecycle...")

    # Find the current Production model and demote it to Staging
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for prod_model in prod_versions:
        logging.info(
            f"Demoting Version {prod_model.version} from Production to Staging."
        )
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=prod_model.version,
            stage="Staging",
            archive_existing_versions=False,
        )

    # Promote the new model to Production
    logging.info(f"Promoting Version {version_num} to Production.")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version_num,
        stage="Production",
        archive_existing_versions=False,
    )

    logging.info("Model registry updated successfully.")


if __name__ == "__main__":
    validate_and_promote()
