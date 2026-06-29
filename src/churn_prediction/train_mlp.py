from datetime import UTC, datetime

import mlflow

from churn_prediction.artifacts import (
    save_feature_names,
    save_metadata,
    save_mlp,
    save_preprocessor,
    save_threshold,
)
from churn_prediction.baselines import evaluate_baselines_on_test
from churn_prediction.config import DATA_PATH, MODEL_VERSION, MODELS_DIR, SEED
from churn_prediction.cost_analysis import best_threshold, threshold_cost_table
from churn_prediction.data import (
    dataset_hash,
    load_churn_csv,
    set_seed,
    split_features_target,
    train_val_test_split,
)
from churn_prediction.evaluate import classification_metrics
from churn_prediction.neural_network import predict_proba, train_mlp
from churn_prediction.preprocessing import fit_transform_preprocessor, get_feature_names


def main() -> None:
    set_seed(SEED)
    df = load_churn_csv(DATA_PATH)
    x, y = split_features_target(df)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)
    preprocessor, x_train_p = fit_transform_preprocessor(x_train)
    x_val_p = preprocessor.transform(x_val)
    x_test_p = preprocessor.transform(x_test)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    with mlflow.start_run(run_name="mlp"):
        result = train_mlp(
            x_train_p, y_train.to_numpy(), x_val_p, y_val.to_numpy(), pos_weight=pos_weight
        )
        val_prob = predict_proba(result.model, x_val_p)
        table = threshold_cost_table(y_val.to_numpy(), val_prob)
        threshold = best_threshold(table)
        test_prob = predict_proba(result.model, x_test_p)
        metrics = classification_metrics(y_test.to_numpy(), test_prob, threshold)
        final_comparison = evaluate_baselines_on_test(x_train, y_train, x_test, y_test)
        final_comparison.loc[len(final_comparison)] = {"model": "PyTorch MLP"} | metrics
        final_comparison.to_csv(MODELS_DIR / "final_model_comparison.csv", index=False)
        mlflow.log_params(
            {
                "architecture": "64-32",
                "seed": SEED,
                "best_epoch": result.best_epoch,
                "pos_weight": pos_weight,
            }
        )
        mlflow.log_metrics(metrics)
        save_preprocessor(preprocessor)
        save_mlp(result.model, x_train_p.shape[1])
        save_feature_names(get_feature_names(preprocessor))
        save_threshold(threshold)
        metadata = {
            "model": "pytorch_mlp",
            "version": MODEL_VERSION,
            "trained_at": datetime.now(UTC).isoformat(),
            "metrics": metrics,
            "seed": SEED,
            "feature_count": int(x_train_p.shape[1]),
            "threshold": threshold,
            "dataset_hash": dataset_hash(DATA_PATH),
        }
        save_metadata(metadata)
        mlflow.log_dict(metadata, "metadata.json")
        mlflow.log_artifact(str(MODELS_DIR / "final_model_comparison.csv"), artifact_path="reports")


if __name__ == "__main__":
    main()
