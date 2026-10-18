from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.features import NUMERIC_FEATURES


@dataclass
class ModelResult:
    pipeline: Pipeline
    metrics: dict


def train_bankruptcy_model(model_df: pd.DataFrame, config: dict) -> ModelResult:
    """Train a simple logistic-regression probability-of-bankruptcy model."""
    X = model_df[NUMERIC_FEATURES]
    y = model_df["bankrupt_within_12m"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, NUMERIC_FEATURES)]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = config["model"]["bankruptcy_probability_threshold"]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    return ModelResult(pipeline=pipeline, metrics=metrics)
