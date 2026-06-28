import json
from pathlib import Path
from typing import Any

import joblib
import torch
from sklearn.compose import ColumnTransformer

from churn_prediction.config import (
    FEATURE_NAMES_PATH,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
)
from churn_prediction.neural_network import ChurnMLP


def save_json(path: Path, data: dict[str, Any] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_preprocessor(preprocessor: ColumnTransformer, path: Path = PREPROCESSOR_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> ColumnTransformer:
    return joblib.load(path)


def save_mlp(model: ChurnMLP, input_size: int, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_size": input_size, "state_dict": model.state_dict()}, path)


def load_mlp(path: Path = MODEL_PATH) -> ChurnMLP:
    checkpoint = torch.load(path, map_location="cpu")
    model = ChurnMLP(int(checkpoint["input_size"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model


def artifacts_ready(models_dir: Path = MODELS_DIR) -> bool:
    required = ["preprocessor.joblib", "model.pt", "metadata.json", "threshold.json"]
    return all((models_dir / name).exists() for name in required)


def save_metadata(metadata: dict[str, Any]) -> None:
    save_json(METADATA_PATH, metadata)


def save_feature_names(names: list[str]) -> None:
    save_json(FEATURE_NAMES_PATH, names)


def save_threshold(threshold: float) -> None:
    save_json(THRESHOLD_PATH, {"threshold": threshold})
