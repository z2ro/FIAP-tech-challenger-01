import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn_prediction.config import SEED
from churn_prediction.preprocessing import build_preprocessor


def baseline_models(seed: int = SEED) -> dict[str, object]:
    return {
        "dummy": DummyClassifier(strategy="prior", random_state=seed),
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed
        ),
    }


def cross_validate_baselines(x: pd.DataFrame, y: pd.Series, seed: int = SEED) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    for name, model in baseline_models(seed).items():
        pipe = Pipeline([("preprocessor", build_preprocessor(x)), ("model", clone(model))])
        scores = cross_validate(pipe, x, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({"model": name} | {m: float(np.mean(scores[f"test_{m}"])) for m in scoring})
    return pd.DataFrame(rows).rename(columns={"average_precision": "pr_auc"})
