import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def save_reports(
    y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path, threshold: float = 0.5
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    pd.DataFrame(confusion_matrix(y_true, y_pred)).to_csv(
        out_dir / "confusion_matrix.csv", index=False
    )
    (out_dir / "classification_report.json").write_text(
        json.dumps(classification_report(y_true, y_pred, output_dict=True), indent=2)
    )


def comparison_table(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["pr_auc", "recall", "f1"], ascending=False)
