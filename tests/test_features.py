import pandas as pd
import numpy as np
import pytest
from features.transformers import DomainFeatureGenerator

def test_domain_feature_generator():
    """Tests the custom transformer for correct calculations and NaN handling."""
    
    # Create dummy data including the 365243 anomaly and potential divide-by-zero
    dummy_data = pd.DataFrame({
        'AMT_INCOME_TOTAL': [100000, 0, 50000],
        'AMT_CREDIT': [200000, 100000, 0],
        'AMT_ANNUITY': [10000, 5000, 0],
        'DAYS_EMPLOYED': [-1000, 365243, -500],
        'DAYS_BIRTH': [-10000, -20000, -15000]
    })
    
    transformer = DomainFeatureGenerator()
    transformed_df = transformer.transform(dummy_data)
    
    # 1. Test Anomaly Replacement
    assert np.isnan(transformed_df.loc[1, 'DAYS_EMPLOYED']), "Anomaly 365243 was not replaced with NaN"
    
    # 2. Test Calculation Logic
    expected_ratio = 10000 / 100000
    assert np.isclose(transformed_df.loc[0, 'INC_TO_ANNUITY_RATIO'], expected_ratio, atol=1e-4)
    
    # 3. Test Divide by Zero Handling (Should not be infinity)
    assert not np.isinf(transformed_df.loc[1, 'INC_TO_ANNUITY_RATIO']), "Divide by zero resulted in Inf"
    assert not np.isinf(transformed_df.loc[2, 'CREDIT_TO_ANNUITY_RATIO']), "Divide by zero resulted in Inf"