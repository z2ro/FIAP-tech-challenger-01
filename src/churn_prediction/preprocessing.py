import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.data import feature_columns


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = feature_columns(x_train)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def fit_transform_preprocessor(x_train: pd.DataFrame) -> tuple[ColumnTransformer, np.ndarray]:
    preprocessor = build_preprocessor(x_train)
    return preprocessor, preprocessor.fit_transform(x_train)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return preprocessor.get_feature_names_out().tolist()
