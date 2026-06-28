from typing import Any

import pandas as pd

from churn_prediction.artifacts import artifact_path, load_json, load_mlp, load_preprocessor
from churn_prediction.neural_network import predict_proba


def risk_level(probability: float) -> str:
    if probability < 0.30:
        return "low"
    if probability < 0.60:
        return "medium"
    return "high"


class ChurnPredictor:
    def __init__(self) -> None:
        self.preprocessor = load_preprocessor()
        self.model = load_mlp()
        self.threshold = float(load_json(artifact_path("threshold")).get("threshold", 0.5))
        self.metadata: dict[str, Any] = load_json(artifact_path("metadata"))

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        x = self.preprocessor.transform(pd.DataFrame([payload]))
        probability = float(predict_proba(self.model, x)[0])
        return {
            "churn_probability": probability,
            "prediction": int(probability >= self.threshold),
            "threshold": self.threshold,
            "risk_level": risk_level(probability),
            "model_version": self.metadata.get("version", "unknown"),
        }
