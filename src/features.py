"""Feature engineering and windowing helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMN = "nat_demand"

WEATHER_COLUMNS = [
    "T2M_toc",
    "QV2M_toc",
    "TQL_toc",
    "W2M_toc",
    "T2M_san",
    "QV2M_san",
    "TQL_san",
    "W2M_san",
    "T2M_dav",
    "QV2M_dav",
    "TQL_dav",
    "W2M_dav",
]

CALENDAR_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "holiday",
    "school",
]

DEFAULT_FEATURE_COLUMNS = [TARGET_COLUMN, *WEATHER_COLUMNS, *CALENDAR_COLUMNS]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features from a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex before creating time features.")

    featured = df.copy()
    hour = featured.index.hour
    dayofweek = featured.index.dayofweek
    month = featured.index.month

    featured["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    featured["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    featured["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    featured["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    featured["month_sin"] = np.sin(2 * np.pi * month / 12)
    featured["month_cos"] = np.cos(2 * np.pi * month / 12)
    featured["is_weekend"] = (dayofweek >= 5).astype(int)
    return featured


def validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one-step-ahead sliding windows."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    if len(X) <= sequence_length:
        raise ValueError("Not enough rows to create at least one sequence.")

    windows = []
    targets = []
    for index in range(sequence_length, len(X)):
        windows.append(X[index - sequence_length : index])
        targets.append(y[index])
    return np.asarray(windows), np.asarray(targets)
