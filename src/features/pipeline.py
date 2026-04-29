import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

from features.transformers import DomainFeatureGenerator

logger = logging.getLogger(__name__)

def build_preprocessor(numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Builds the sklearn preprocessing pipeline.
    """
    logger.info("Building preprocessing pipeline...")
    
    # Numeric Strategy: Median imputation -> Standard Scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Strategy: Constant imputation -> One-Hot Encoding
    # sparse_output=False is crucial for SHAP compatibility later
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop columns not explicitly declared
    )

    # Full pipeline includes domain feature generation first
    full_pipeline = Pipeline(steps=[
        ('domain_features', DomainFeatureGenerator()),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline