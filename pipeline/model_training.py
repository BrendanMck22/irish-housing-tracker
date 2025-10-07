"""Model training utilities for the Irish house price gap pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class TrainingConfig:
    test_size: float = 0.3
    random_state: int = 42
    n_estimators: int = 500


@dataclass
class TrainingResult:
    model: RandomForestRegressor
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: pd.Series
    mae_total: float
    r2_total: float
    metrics_per_county_year: pd.DataFrame
    input_example: pd.DataFrame
    signature: object


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    counties: pd.DataFrame,
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train the random forest baseline model and compute evaluation metrics."""
    cfg = config or TrainingConfig()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=counties
    )

    model = RandomForestRegressor(n_estimators=cfg.n_estimators, random_state=cfg.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae_total = mean_absolute_error(y_test, y_pred)
    r2_total = r2_score(y_test, y_pred)

    X_test_eval = X_test.copy()
    X_test_eval['y_true'] = y_test.values
    X_test_eval['y_pred'] = y_pred

    county_cols: List[str] = [col for col in X_test_eval.columns if col.startswith('county_')]
    if county_cols:
        X_test_eval['county'] = (
            X_test_eval[county_cols]
            .idxmax(axis=1)
            .str.replace('county_', '', regex=False)
        )
    else:
        X_test_eval['county'] = 'unknown'

    metrics_per_county_year = (
        X_test_eval.groupby(['county', 'year'])
        .apply(
            lambda g: pd.Series(
                {
                    'MAE': mean_absolute_error(g['y_true'], g['y_pred']),
                    'R2': r2_score(g['y_true'], g['y_pred']) if len(g) > 1 else 0.0,
                }
            )
        )
        .reset_index()
    )

    input_example = X_test.head(1)
    signature = infer_signature(X_test, y_pred)

    return TrainingResult(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=pd.Series(y_pred, index=y_test.index),
        mae_total=mae_total,
        r2_total=r2_total,
        metrics_per_county_year=metrics_per_county_year,
        input_example=input_example,
        signature=signature,
    )
