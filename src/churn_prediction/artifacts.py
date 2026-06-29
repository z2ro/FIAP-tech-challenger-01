import json
import os
from pathlib import Path
from typing import Any

import joblib
import torch
from sklearn.compose import ColumnTransformer

from churn_prediction.config import MODELS_DIR
from churn_prediction.neural_network import ChurnMLP

_ARTIFACT_FILES = {
    "preprocessor": "preprocessor.joblib",
    "model": "model.pt",
    "metadata": "metadata.json",
    "feature_names": "feature_names.json",
    "threshold": "threshold.json",
}


def models_dir() -> Path:
    return Path(os.getenv("CHURN_MODELS_DIR", str(MODELS_DIR)))


def artifact_path(name: str) -> Path:
    return models_dir() / _ARTIFACT_FILES[name]


def save_json(path: Path, data: dict[str, Any] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_preprocessor(preprocessor: ColumnTransformer, path: Path | None = None) -> None:
    path = path or artifact_path("preprocessor")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path | None = None) -> ColumnTransformer:
    return joblib.load(path or artifact_path("preprocessor"))


def save_mlp(model: ChurnMLP, input_size: int, path: Path | None = None) -> None:
    path = path or artifact_path("model")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"input_size": input_size, "state_dict": model.state_dict()}, tmp)
    tmp.replace(path)


def load_mlp(path: Path | None = None) -> ChurnMLP:
    checkpoint = torch.load(path or artifact_path("model"), map_location="cpu")
    model = ChurnMLP(int(checkpoint["input_size"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model


def artifacts_ready(directory: Path | None = None) -> bool:
    directory = directory or models_dir()
    required = ["preprocessor.joblib", "model.pt", "metadata.json", "threshold.json"]
    return all((directory / name).exists() for name in required)


def save_metadata(metadata: dict[str, Any]) -> None:
    save_json(artifact_path("metadata"), metadata)


def save_feature_names(names: list[str]) -> None:
    save_json(artifact_path("feature_names"), names)


def save_threshold(threshold: float) -> None:
    save_json(artifact_path("threshold"), {"threshold": threshold})
