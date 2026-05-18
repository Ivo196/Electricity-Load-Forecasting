"""Data loading, chronological splitting, scaling, and loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from .features import (
    DEFAULT_FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_time_features,
    create_sequences,
    validate_columns,
)


@dataclass(frozen=True)
class SplitConfig:
    train_size: float = 0.7
    validation_size: float = 0.15
    test_size: float = 0.15

    def validate(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if not np.isclose(total, 1.0):
            raise ValueError("train_size + validation_size + test_size must equal 1.0.")
        if min(self.train_size, self.validation_size, self.test_size) <= 0:
            raise ValueError("All split sizes must be positive.")


@dataclass
class PreparedData:
    feature_columns: list[str]
    target_column: str
    sequence_length: int
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    frames: dict[str, pd.DataFrame]
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    return df


def split_chronologically(
    df: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> dict[str, pd.DataFrame]:
    config.validate()
    n_rows = len(df)
    train_end = int(n_rows * config.train_size)
    validation_end = train_end + int(n_rows * config.validation_size)

    return {
        "train": df.iloc[:train_end].copy(),
        "validation": df.iloc[train_end:validation_end].copy(),
        "test": df.iloc[validation_end:].copy(),
    }


def prepare_data(
    data_path: str | Path,
    sequence_length: int = 24 * 7,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
    split_config: SplitConfig = SplitConfig(),
) -> PreparedData:
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS

    df = add_time_features(load_dataset(data_path))
    validate_columns(df, [*feature_columns, target_column])
    model_columns = list(dict.fromkeys([*feature_columns, target_column]))
    model_df = df[model_columns].copy()

    frames = split_chronologically(model_df, split_config)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_raw = frames["train"][feature_columns].to_numpy()
    y_train_raw = frames["train"][[target_column]].to_numpy()

    feature_scaler.fit(X_train_raw)
    target_scaler.fit(y_train_raw)

    scaled = {}
    for split_name, split_frame in frames.items():
        X_scaled = feature_scaler.transform(split_frame[feature_columns].to_numpy())
        y_scaled = target_scaler.transform(split_frame[[target_column]].to_numpy())
        scaled[split_name] = create_sequences(X_scaled, y_scaled, sequence_length)

    return PreparedData(
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        frames=frames,
        X_train=scaled["train"][0],
        y_train=scaled["train"][1],
        X_validation=scaled["validation"][0],
        y_validation=scaled["validation"][1],
        X_test=scaled["test"][0],
        y_test=scaled["test"][1],
    )


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_loaders(
    prepared: PreparedData,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        make_loader(prepared.X_train, prepared.y_train, batch_size, shuffle=True),
        make_loader(prepared.X_validation, prepared.y_validation, batch_size),
        make_loader(prepared.X_test, prepared.y_test, batch_size),
    )
