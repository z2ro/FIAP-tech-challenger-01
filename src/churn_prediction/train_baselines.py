import json
import tempfile

import joblib
import mlflow
from sklearn.pipeline import Pipeline

from churn_prediction.artifacts import save_json
from churn_prediction.baselines import baseline_models, cross_validate_baselines
from churn_prediction.config import BASELINE_MODEL_PATH, DATA_PATH, MODELS_DIR, SEED
from churn_prediction.data import dataset_hash, load_churn_csv, set_seed, split_features_target
from churn_prediction.preprocessing import build_preprocessor


def main() -> None:
    set_seed(SEED)
    df = load_churn_csv(DATA_PATH)
    x, y = split_features_target(df)
    scores = cross_validate_baselines(x, y)
    scores.to_csv(MODELS_DIR / "baseline_comparison.csv", index=False)
    best_name = scores.sort_values(["pr_auc", "recall", "f1"], ascending=False).iloc[0]["model"]
    pipe = Pipeline(
        [("preprocessor", build_preprocessor(x)), ("model", baseline_models()[str(best_name)])]
    )
    with mlflow.start_run(run_name=f"baseline_{best_name}"):
        mlflow.log_param("model", best_name)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("dataset_hash", dataset_hash(DATA_PATH))
        for _, row in scores.iterrows():
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
                mlflow.log_metric(f"{row['model']}_{metric}", float(row[metric]))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(scores.to_dict(orient="records"), f, indent=2)
            mlflow.log_artifact(f.name, artifact_path="reports")
        pipe.fit(x, y)
        joblib.dump(pipe, BASELINE_MODEL_PATH)
        mlflow.log_artifact(str(BASELINE_MODEL_PATH), artifact_path="models")
    save_json(MODELS_DIR / "baseline_metadata.json", {"selected_model": str(best_name)})


if __name__ == "__main__":
    main()
