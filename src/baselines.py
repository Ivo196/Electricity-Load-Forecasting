"""Baseline models for one-step electricity demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .features import DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class MetricResult:
    model: str
    mae: float
    rmse: float
    mape: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> MetricResult:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    return MetricResult(model=model_name, mae=mae, rmse=rmse, mape=mape)


def evaluate_lag_baseline(
    frame: pd.DataFrame,
    lag_hours: int,
    model_name: str,
    target_column: str = TARGET_COLUMN,
    start_offset: int = 24 * 7,
) -> MetricResult:
    y_true = frame[target_column].iloc[start_offset:]
    y_pred = frame[target_column].shift(lag_hours).iloc[start_offset:]
    valid = y_pred.notna()
    return regression_metrics(y_true[valid].to_numpy(), y_pred[valid].to_numpy(), model_name)


def make_tabular_frame(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    lags: tuple[int, ...] = (1, 24, 168),
) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    engineered = frame.copy()
    lag_columns = []
    for lag in lags:
        column = f"{target_column}_lag_{lag}"
        engineered[column] = engineered[target_column].shift(lag)
        lag_columns.append(column)

    engineered[f"{target_column}_rolling_24_mean"] = (
        engineered[target_column].shift(1).rolling(window=24).mean()
    )
    engineered[f"{target_column}_rolling_168_mean"] = (
        engineered[target_column].shift(1).rolling(window=168).mean()
    )

    model_features = [
        column for column in feature_columns if column != target_column
    ] + lag_columns + [
        f"{target_column}_rolling_24_mean",
        f"{target_column}_rolling_168_mean",
    ]

    supervised = engineered.dropna(subset=[*model_features, target_column])
    return supervised[model_features], supervised[target_column]


def evaluate_tabular_baselines(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    include_xgboost: bool = True,
) -> list[MetricResult]:
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    X_train, y_train = make_tabular_frame(train_frame, target_column, feature_columns)
    X_test, y_test = make_tabular_frame(test_frame, target_column, feature_columns)

    models = [
        ("linear_regression", LinearRegression()),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    if include_xgboost:
        try:
            from xgboost import XGBRegressor

            models.append(
                (
                    "xgboost",
                    XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            )
        except ImportError:
            pass

    results = []
    for model_name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results.append(regression_metrics(y_test.to_numpy(), predictions, model_name))
    return results
