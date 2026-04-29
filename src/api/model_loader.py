import mlflow
import logging
import joblib
import sys
from pathlib import Path

# Add 'src/' to Python path so joblib can find the 'features' module
# when unpickling the preprocessing pipeline's custom transformers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def load_production_assets():
    """
    Loads the current Production model from MLflow Registry
    and the local preprocessing pipeline.
    """
    # Point to the local tracking DB
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Load Preprocessing Pipeline
    logging.info("Loading preprocessing pipeline...")
    pipeline_path = "models/preprocessing_pipeline.joblib"
    try:
        pipeline = joblib.load(pipeline_path)
    except FileNotFoundError:
        logging.error(
            f"Pipeline not found at {pipeline_path}. Run feature engineering first."
        )
        raise

    # Load Top Features List
    logging.info("Loading top features list...")
    features_path = "models/top_features.joblib"
    try:
        top_features = joblib.load(features_path)
    except FileNotFoundError:
        logging.error(f"Features list not found at {features_path}.")
        raise

    # Load Production Model from MLflow
    logging.info("Fetching Production model from MLflow Registry...")
    model_uri = "models:/CreditRiskModel/Production"

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info("Production model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model from MLflow: {e}")
        raise

    return pipeline, top_features, model


if __name__ == "__main__":
    # Test the loader
    pipe, feats, prod_model = load_production_assets()
    print(f"Loaded model type: {type(prod_model)}")
    print(f"Number of expected features: {len(feats)}")
