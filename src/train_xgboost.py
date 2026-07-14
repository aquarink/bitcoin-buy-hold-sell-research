from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .baselines import CLASS_ORDER


LABEL_TO_INT = {label: idx for idx, label in enumerate(CLASS_ORDER)}
INT_TO_LABEL = {idx: label for label, idx in LABEL_TO_INT.items()}


@dataclass
class XGBoostFitResult:
    model: XGBClassifier
    params: dict
    validation_macro_f1: float


def encode_labels(labels: pd.Series) -> np.ndarray:
    return labels.map(LABEL_TO_INT).to_numpy()


def decode_labels(values: np.ndarray) -> pd.Series:
    return pd.Series(values).map(INT_TO_LABEL)


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    param_grid: list[dict],
    seed: int = 42,
) -> XGBoostFitResult:
    from sklearn.metrics import f1_score

    y_train_enc = encode_labels(y_train)
    y_valid_enc = encode_labels(y_valid)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    best_result: XGBoostFitResult | None = None
    for params in param_grid:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_ORDER),
            random_state=seed,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=4,
            **params,
        )
        model.fit(
            X_train,
            y_train_enc,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid_enc)],
            verbose=False,
        )
        valid_pred = model.predict(X_valid)
        macro_f1 = float(f1_score(y_valid_enc, valid_pred, average="macro"))
        if best_result is None or macro_f1 > best_result.validation_macro_f1:
            best_result = XGBoostFitResult(model=model, params=params, validation_macro_f1=macro_f1)

    if best_result is None:
        raise RuntimeError("XGBoost tuning produced no result.")
    return best_result
