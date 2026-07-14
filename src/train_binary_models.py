from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


@dataclass
class BinaryFitResult:
    model: object
    validation_score: float | None = None
    params: dict | None = None


def fit_binary_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
) -> BinaryFitResult:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return BinaryFitResult(model=model)


def fit_binary_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    param_grid: list[dict],
    seed: int = 42,
) -> BinaryFitResult:
    from sklearn.metrics import f1_score

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    best: BinaryFitResult | None = None
    for params in param_grid:
        model = XGBClassifier(
            objective="binary:logistic",
            random_state=seed,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=4,
            **params,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        pred = (model.predict_proba(X_valid)[:, 1] >= 0.5).astype(int)
        score = float(f1_score(y_valid, pred, zero_division=0))
        if best is None or score > (best.validation_score or -1):
            best = BinaryFitResult(model=model, validation_score=score, params=params)
    if best is None:
        raise RuntimeError("Binary XGBoost tuning produced no result.")
    return best
