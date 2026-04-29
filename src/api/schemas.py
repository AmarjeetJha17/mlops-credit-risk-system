from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional


class LoanApplication(BaseModel):
    """
    Pydantic v2 Schema for incoming loan applications.
    We define the key features required for our custom domain transformer,
    and allow extra fields for the raw categorical/numerical features.
    """

    model_config = ConfigDict(extra="allow")  # Allows dynamic features

    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Total income of the client")
    AMT_CREDIT: float = Field(..., gt=0, description="Credit amount of the loan")
    AMT_ANNUITY: float = Field(..., gt=0, description="Loan annuity")
    DAYS_EMPLOYED: int = Field(
        ..., description="Days employed (negative values, 365243 for unemployed)"
    )
    DAYS_BIRTH: int = Field(
        ..., description="Client's age in days at the time of application (negative)"
    )

    # Add a few common categorical fields for completeness
    NAME_CONTRACT_TYPE: str = Field(default="Cash loans", description="Type of loan")
    CODE_GENDER: str = Field(default="M", description="Gender of the client")
    FLAG_OWN_CAR: str = Field(default="N", description="Does the client own a car?")
    FLAG_OWN_REALTY: str = Field(
        default="Y", description="Does the client own a house?"
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Repaid, 1 = Default")
    probability: float = Field(..., description="Probability of default")
    feature_contributions: Optional[Dict[str, float]] = Field(
        default=None, description="SHAP values for explainability"
    )
    model_version: str = Field(
        ..., description="Version of the model serving this request"
    )


class MetricsResponse(BaseModel):
    model_name: str
    version: str
    stage: str
    training_date: str
    validation_roc_auc: str
