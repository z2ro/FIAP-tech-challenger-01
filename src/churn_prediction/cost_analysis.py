import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def threshold_cost_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
    contact_cost: float = 10.0,
    expected_churn_loss: float = 100.0,
    retention_success_probability: float = 0.2,
) -> pd.DataFrame:
    thresholds = thresholds or [round(x, 2) for x in np.linspace(0.1, 0.9, 17)]
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        benefit = tp * expected_churn_loss * retention_success_probability
        cost = (tp + fp) * contact_cost + fn * expected_churn_loss
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "net_benefit": benefit - cost,
            }
        )
    return pd.DataFrame(rows)


def best_threshold(cost_table: pd.DataFrame) -> float:
    return float(cost_table.sort_values("net_benefit", ascending=False).iloc[0]["threshold"])
