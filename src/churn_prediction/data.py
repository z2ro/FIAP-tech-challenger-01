import hashlib
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from churn_prediction.config import SEED, TARGET


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def snake_case(name: str) -> str:
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def load_churn_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. "
            "Baixe o IBM Telco Customer Churn e salve neste caminho."
        )
    return clean_churn_dataframe(pd.read_csv(path))


def clean_churn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [snake_case(c) for c in out.columns]
    if "customer_id" in out.columns:
        out = out.drop(columns=["customer_id"])
    if "total_charges" in out.columns:
        out["total_charges"] = pd.to_numeric(out["total_charges"], errors="coerce")
    if TARGET in out.columns:
        out[TARGET] = out[TARGET].map({"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0})
    return out


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": df.shape,
        "missing": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "cardinality": df.nunique(dropna=True).to_dict(),
        "target_distribution": df[TARGET].value_counts(normalize=True).to_dict()
        if TARGET in df
        else {},
        "numeric_describe": df.select_dtypes(include="number").describe().to_dict(),
    }


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET].astype(int)


def train_val_test_split(
    x: pd.DataFrame, y: pd.Series, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def feature_columns(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    return numeric, categorical
