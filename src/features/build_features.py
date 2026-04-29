import pandas as pd
import numpy as np
import joblib
import logging
import shap
import lightgbm as lgb
import os
import re

from features.pipeline import build_preprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RAW_DATA_PATH = "data/raw/application_train.csv"
PIPELINE_PATH = "models/preprocessing_pipeline.joblib"
TOP_FEATURES_PATH = "models/top_features.joblib"

def get_feature_lists(df: pd.DataFrame):
    """Separates numeric and categorical columns, ignoring ID and Target."""
    exclude_cols = ['SK_ID_CURR', 'TARGET']
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Identify domains created by custom transformer to include in numeric list
    domain_cols = ['INC_TO_ANNUITY_RATIO', 'CREDIT_TO_ANNUITY_RATIO', 
                   'CREDIT_TO_INCOME_RATIO', 'EMPLOYED_TO_AGE_RATIO']
    
    numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features.extend(domain_cols)
    
    categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numeric_features, categorical_features

def main():
    logging.info("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    y = df['TARGET']
    X = df.drop(columns=['TARGET'])

    numeric_features, categorical_features = get_feature_lists(df)
    
    pipeline = build_preprocessor(numeric_features, categorical_features)
    
    logging.info("Fitting preprocessing pipeline (This may take a minute)...")
    # Fit and transform data
    X_transformed_array = pipeline.fit_transform(X)
    
    # Extract feature names after OneHotEncoding
    raw_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # MLOps Fix: LightGBM crashes if feature names contain special JSON characters.
    # We use regex to replace anything that isn't a letter, number, or underscore with an underscore.
    clean_feature_names = [re.sub(r'[^A-Za-z0-9_]+', '_', name) for name in raw_feature_names]
    
    X_transformed = pd.DataFrame(X_transformed_array, columns=clean_feature_names)
    
    logging.info(f"Data transformed. Shape: {X_transformed.shape}")

    # Handling class imbalance using pos_scale_weight for the SHAP evaluation model
    imbalance_ratio = (y == 0).sum() / (y == 1).sum()
    
    logging.info("Training temporary LightGBM model for SHAP evaluation...")
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        scale_pos_weight=imbalance_ratio, # Handles class imbalance
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_transformed, y)

    logging.info("Calculating SHAP values...")
    # Use TreeExplainer for tree-based models (fastest)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)
    
    # LightGBM binary classification returns SHAP values as a list [class_0, class_1]. 
    # We want class_1 (Default). If it's an array, it's already class 1.
    if isinstance(shap_values, list):
        shap_values_target = shap_values[1]
    else:
        shap_values_target = shap_values

    # Calculate mean absolute SHAP values for feature importance
    mean_shap = np.abs(shap_values_target).mean(axis=0)
    shap_df = pd.DataFrame({'feature': clean_feature_names, 'shap_importance': mean_shap})
    shap_df = shap_df.sort_values(by='shap_importance', ascending=False)
    
    top_50_features = shap_df.head(50)['feature'].tolist()
    logging.info(f"Top 5 Features by SHAP: {top_50_features[:5]}")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump(top_50_features, TOP_FEATURES_PATH)
    logging.info(f"Pipeline saved to {PIPELINE_PATH}")
    logging.info(f"Top features saved to {TOP_FEATURES_PATH}")

if __name__ == "__main__":
    main()