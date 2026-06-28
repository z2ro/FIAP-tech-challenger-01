from typing import Literal

import pandera.pandas as pa
from pydantic import BaseModel, ConfigDict, Field

ChurnDataFrameSchema = pa.DataFrameSchema({"churn": pa.Column(int, pa.Check.isin([0, 1]))})


class CustomerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    senior_citizen: int = Field(ge=0, le=1)
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    phone_service: Literal["Yes", "No"]
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: Literal["Yes", "No"]
    payment_method: str
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int
    threshold: float
    risk_level: Literal["low", "medium", "high"]
    model_version: str
