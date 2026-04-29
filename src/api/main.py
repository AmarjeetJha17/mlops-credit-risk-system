import os
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        assets['pipeline'] = joblib.load("models/preprocessing_pipeline.joblib")
        assets['top_features'] = joblib.load("models/top_features.joblib")
        
        # Load MLflow Model
        client = MlflowClient()
        # Fetch metadata for the /metrics endpoint (using modern API)
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == stage]
        if not prod_versions:
            raise RuntimeError(f"No model version found in '{stage}' stage.")
        model_metadata = client.get_latest_versions(model_name, stages=[stage])[0]
        assets['metadata'] = {
            "version": str(model_metadata.version),  # <--- FIX: Cast integer to string
            "roc_auc": model_metadata.tags.get("roc_auc", "unknown"),
            "training_date": model_metadata.tags.get("training_date", "unknown")
        }
        
        # Load the underlying scikit-learn/LightGBM model (better for SHAP support)
        # In Docker, the MLflow DB may contain hardcoded Windows artifact paths
        # that don't resolve on Linux. Detect this and load from the local path.
        is_docker = os.path.exists("/.dockerenv")
        if is_docker:
            # The model source in the registry is like "models:/m-<hash>"
            # which maps to mlruns/<exp_id>/models/<model_hash>/artifacts/
            model_source = model_metadata.source
            run_id = model_metadata.run_id
            # Get experiment_id from the run
            run_info = client.get_run(run_id)
            exp_id = run_info.info.experiment_id
            # Extract model hash from source (format: "models:/m-<hash>")
            model_hash = model_source.split("/")[-1]
            local_model_path = f"mlruns/{exp_id}/models/{model_hash}/artifacts"
            logger.info(f"Docker detected: loading model from local path: {local_model_path}")
            assets['model'] = mlflow.sklearn.load_model(local_model_path)
        else:
            model_uri = f"models:/{model_name}/{stage}"
            assets['model'] = mlflow.sklearn.load_model(model_uri)
        
        # Initialize SHAP explainer
        logger.info("Initializing SHAP explainer...")
        assets['explainer'] = shap.TreeExplainer(assets['model'])
        
        logger.info(f"Model V{model_metadata.version} loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load assets: {e}")
        # In a real system, you might load a local fallback model here
        raise RuntimeError("Could not load production model. API cannot start.")
        
    yield # API is running
    
    # Clean up on shutdown
    logger.info("Shutting down API and clearing memory...")
    assets.clear()

# Initialize FastAPI
app = FastAPI(
    title="MLOps Credit Risk API",
    version="1.0.0",
    description="Real-time loan default prediction service",
    lifespan=lifespan
)

# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Incoming request {request_id}: {request.method} {request.url}")
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Completed request {request_id}: Status {response.status_code} in {process_time:.2f}ms")
    return response

@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe for Docker/Kubernetes."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Returns the performance metrics of the currently loaded production model."""
    if 'metadata' not in assets:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
        
    return MetricsResponse(
        model_name="CreditRiskModel",
        version=assets['metadata']['version'],
        stage="Production",
        training_date=assets['metadata']['training_date'],
        validation_roc_auc=assets['metadata']['roc_auc']
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(application: LoanApplication):
    """Processes a loan application and returns a default probability."""
    try:
        # 1. Convert Pydantic payload to DataFrame
        # Using model_dump() for Pydantic v2
        df_input = pd.DataFrame([application.model_dump()])
        
        # 2. Pad missing columns with NaN
        # The pipeline was fitted on the full Home Credit dataset (~125 columns).
        # The API only receives a subset; the pipeline's imputers handle the rest.
        ct = assets['pipeline'].named_steps['preprocessor']
        expected_cols = []
        for name, trans, cols in ct.transformers_:
            expected_cols.extend(cols)
        missing_cols = [col for col in expected_cols if col not in df_input.columns]
        if missing_cols:
            missing_df = pd.DataFrame(np.nan, index=df_input.index, columns=missing_cols)
            df_input = pd.concat([df_input, missing_df], axis=1)
        
        # 3. Transform through the pre-fitted pipeline
        X_transformed_array = assets['pipeline'].transform(df_input)
        raw_feature_names = assets['pipeline'].named_steps['preprocessor'].get_feature_names_out()
        
        # Apply the exact same regex sanitization from training
        clean_feature_names = [re.sub(r'[^A-Za-z0-9_]+', '_', name) for name in raw_feature_names]
        X_transformed = pd.DataFrame(X_transformed_array, columns=clean_feature_names)
        
        # Filter down to top features
        X_final = X_transformed[assets['top_features']]
        
        # 3. Predict
        probability = float(assets['model'].predict_proba(X_final)[0, 1])
        # Threshold tuning: 0.5 is default, but banks might use 0.15 depending on cost matrix
        prediction = 1 if probability > 0.15 else 0 
        
        # 4. Calculate local SHAP explainability
        shap_values = assets['explainer'].shap_values(X_final)
        shap_target = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        # Create a dictionary of the top 5 most impactful features for this specific prediction
        contributions = dict(zip(X_final.columns, shap_target[0]))
        sorted_contributions = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:5])

        # 5. Return Response
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            feature_contributions=sorted_contributions,
            model_version=assets['metadata']['version']
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")