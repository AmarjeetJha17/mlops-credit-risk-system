import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

class DomainFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates domain-specific financial features for credit risk.
    Compatible with sklearn Pipeline.
    """
    def __init__(self):
        # We define variables we expect to exist in the incoming dataframe
        self.expected_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_BIRTH']

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # Fit does nothing here, but must return self for pipeline compatibility
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating domain features...")
        X_out = X.copy()
        
        # Handle Home Credit anomaly: DAYS_EMPLOYED == 365243 means unemployed
        X_out['DAYS_EMPLOYED'] = X_out['DAYS_EMPLOYED'].replace(365243, np.nan)

        # 1. Debt-to-Income Proxy: Percentage of income going to loan annuity
        X_out['INC_TO_ANNUITY_RATIO'] = X_out['AMT_ANNUITY'] / (X_out['AMT_INCOME_TOTAL'] + 1e-5)
        
        # 2. Credit Utilization / Loan Term Proxy: How many years to pay off the loan
        X_out['CREDIT_TO_ANNUITY_RATIO'] = X_out['AMT_CREDIT'] / (X_out['AMT_ANNUITY'] + 1e-5)
        
        # 3. Credit Multiplier: How much more are they borrowing than they make
        X_out['CREDIT_TO_INCOME_RATIO'] = X_out['AMT_CREDIT'] / (X_out['AMT_INCOME_TOTAL'] + 1e-5)
        
        # 4. Employment Stability: Percentage of life spent at current job
        X_out['EMPLOYED_TO_AGE_RATIO'] = X_out['DAYS_EMPLOYED'] / (X_out['DAYS_BIRTH'] - 1e-5)

        # Handle any division by zero infinities created
        X_out.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return X_out