import json

import joblib
import pandas as pd
import pytest

from churn_prediction.artifacts import save_mlp
from churn_prediction.config import METADATA_PATH, MODELS_DIR, PREPROCESSOR_PATH, THRESHOLD_PATH
from churn_prediction.neural_network import ChurnMLP
from churn_prediction.preprocessing import fit_transform_preprocessor


@pytest.fixture
def sample_payload() -> dict[str, object]:
    return {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 70.0,
        "total_charges": 840.0,
    }


@pytest.fixture
def sample_frame(sample_payload: dict[str, object]) -> pd.DataFrame:
    other = sample_payload | {
        "gender": "Male",
        "partner": "No",
        "contract": "One year",
        "monthly_charges": 50.0,
        "total_charges": 600.0,
    }
    return pd.DataFrame([sample_payload, other])


@pytest.fixture
def test_artifacts(sample_frame: pd.DataFrame):
    preprocessor, x = fit_transform_preprocessor(sample_frame)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    model = ChurnMLP(x.shape[1])
    save_mlp(model, x.shape[1])
    METADATA_PATH.write_text(json.dumps({"version": "1.0.0"}))
    THRESHOLD_PATH.write_text(json.dumps({"threshold": 0.5}))
    yield
