import sys
import time
import uuid
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import shap
import re
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager

# Add 'src/' to Python path so joblib can find the 'features' module
# when unpickling the preprocessing pipeline's custom transformers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.schemas import LoanApplication, PredictionResponse, MetricsResponse

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")

# Global variables to hold models in memory
assets = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the production model, pipeline, and features on startup."""
    logger.info("Initializing API and loading model assets...")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_name = "CreditRiskModel"
    stage = "Production"

    try:
        # Load local preprocessing assets
        assets["pipeline"] = joblib.load("models/preprocessing_pipeline.joblib")
        assets["top_features"] = joblib.load("models/top_features.joblib")

        # Load MLflow Metadata
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == stage]
        if not prod_versions:
            raise RuntimeError(f"No model version found in '{stage}' stage.")
        
        model_metadata = prod_versions[0]
        assets["metadata"] = {
            "version": str(model_metadata.version),
            "roc_auc": model_metadata.tags.get("roc_auc", "unknown"),
            "training_date": model_metadata.tags.get("training_date", "unknown"),
        }

        # 1. Load Production (Champion) Model
        prod_uri = f"models:/{model_name}/Production"
        try:
            assets["model_prod"] = mlflow.sklearn.load_model(prod_uri)
        except OSError:
            logger.info("Production registry path failed. Attempting local fallback...")
            assets["model_prod"] = mlflow.sklearn.load_model(model_metadata.source)

        # 2. Load Staging (Challenger) for Shadow Deployment
        try:
            staging_uri = f"models:/{model_name}/Staging"
            assets["model_staging"] = mlflow.sklearn.load_model(staging_uri)
            logger.info("Challenger (Staging) model loaded for Shadow mode.")
        except Exception:
            logger.warning("No Staging model found. Shadow deployment disabled.")
            assets["model_staging"] = None

        # 3. Initialize SHAP explainer on Production model
        logger.info("Initializing SHAP explainer on Champion model...")
        assets["explainer"] = shap.TreeExplainer(assets["model_prod"])

        logger.info(f"Model V{model_metadata.version} and shadow assets initialized.")

    except Exception as e:
        logger.error(f"Failed to load assets: {e}")
        raise RuntimeError("Could not load production model. API cannot start.")

    yield  # API is running

    # Clean up on shutdown
    logger.info("Shutting down API and clearing memory...")
    assets.clear()


# Initialize FastAPI
app = FastAPI(
    title="MLOps Credit Risk API",
    version="1.0.0",
    description="Real-time loan default prediction service",
    lifespan=lifespan,
)


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"Incoming request {request_id}: {request.method} {request.url}")
    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"Completed request {request_id}: Status {response.status_code} in {process_time:.2f}ms"
    )
    return response


@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe for Docker/Kubernetes."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Returns the performance metrics of the currently loaded production model."""
    if "metadata" not in assets:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")

    return MetricsResponse(
        model_name="CreditRiskModel",
        version=assets["metadata"]["version"],
        stage="Production",
        training_date=assets["metadata"]["training_date"],
        validation_roc_auc=assets["metadata"]["roc_auc"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(application: LoanApplication, request: Request):
    """Processes a loan application and returns a default probability."""
    try:
        # 1. Convert Pydantic payload to DataFrame
        # Using model_dump() for Pydantic v2
        df_input = pd.DataFrame([application.model_dump()])

        # 2. Pad missing columns with NaN
        # The pipeline was fitted on the full Home Credit dataset (~125 columns).
        # The API only receives a subset; the pipeline's imputers handle the rest.
        ct = assets["pipeline"].named_steps["preprocessor"]
        expected_cols = []
        for name, trans, cols in ct.transformers_:
            expected_cols.extend(cols)
        missing_cols = [col for col in expected_cols if col not in df_input.columns]
        if missing_cols:
            missing_df = pd.DataFrame(
                np.nan, index=df_input.index, columns=missing_cols
            )
            df_input = pd.concat([df_input, missing_df], axis=1)

        # 3. Transform through the pre-fitted pipeline
        X_transformed_array = assets["pipeline"].transform(df_input)
        raw_feature_names = (
            assets["pipeline"].named_steps["preprocessor"].get_feature_names_out()
        )

        # Apply the exact same regex sanitization from training
        clean_feature_names = [
            re.sub(r"[^A-Za-z0-9_]+", "_", name) for name in raw_feature_names
        ]
        X_transformed = pd.DataFrame(X_transformed_array, columns=clean_feature_names)

        # Filter down to top features
        X_final = X_transformed[assets["top_features"]]

        # 3. Predict with Champion
        prod_prob = float(assets['model_prod'].predict_proba(X_final)[0, 1])
        prediction = 1 if prod_prob > 0.15 else 0 
        
        # 4. Shadow Deployment Logging
        staging_prob = None
        if assets.get('model_staging') is not None:
            try:
                staging_prob = float(assets['model_staging'].predict_proba(X_final)[0, 1])
                # Log both predictions for Evidently AI / Monitoring DB to pick up later
                logger.info(
                    f"SHADOW_LOG | RequestID: {request.headers.get('X-Request-ID', 'unknown')} | "
                    f"Champion_Prob: {prod_prob:.4f} | Challenger_Prob: {staging_prob:.4f}"
                )
            except Exception as e:
                logger.error(f"Shadow prediction failed: {e}")

        # 5. Calculate local SHAP explainability
        shap_values = assets["explainer"].shap_values(X_final)
        shap_target = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Create a dictionary of the top 5 most impactful features for this specific prediction
        contributions = dict(zip(X_final.columns, shap_target[0]))
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[
                :5
            ]
        )

        # 5. Return Response
        return PredictionResponse(
            prediction=prediction,
            probability=prod_prob,
            feature_contributions=sorted_contributions,
            model_version=assets["metadata"]["version"],
        )

    except Exception as e:
        import traceback

        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")
