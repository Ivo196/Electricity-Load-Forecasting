"""Evaluate baselines and the trained LSTM on the final test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .baselines import (
    MetricResult,
    evaluate_lag_baseline,
    evaluate_tabular_baselines,
    regression_metrics,
)
from .data import SplitConfig, make_loader, prepare_data
from .models import LSTMForecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/raw/continuous_dataset.csv")
    parser.add_argument("--model-path", default="models/lstm_forecast.pt")
    parser.add_argument("--metrics-path", default="reports/metrics.csv")
    parser.add_argument("--figure-path", default="reports/figures/lstm_test_forecast.png")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=24 * 7)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--plot-points", type=int, default=500)
    parser.add_argument("--skip-xgboost", action="store_true")
    return parser.parse_args()


def predict_lstm(
    model: LSTMForecast,
    X_test,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    loader = make_loader(X_test, X_test[:, -1, :1], batch_size=batch_size)
    predictions = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in loader:
            predictions.append(model(X_batch.to(device)).cpu())
    return torch.cat(predictions, dim=0)


def load_model(checkpoint_path: Path, device: torch.device) -> LSTMForecast:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LSTMForecast(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_forecast_plot(y_true, y_pred, figure_path: Path, plot_points: int) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    n_points = min(plot_points, len(y_true))
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:n_points], label="Actual", linewidth=1.5)
    plt.plot(y_pred[:n_points], label="LSTM forecast", linewidth=1.5)
    plt.title("Final test split: actual vs LSTM forecast")
    plt.xlabel("Hourly test observations")
    plt.ylabel("National demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    split_config = SplitConfig(
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    prepared = prepare_data(
        args.data_path,
        sequence_length=args.sequence_length,
        split_config=split_config,
    )

    test_frame = prepared.frames["test"]
    results: list[MetricResult] = [
        evaluate_lag_baseline(
            test_frame,
            lag_hours=24,
            model_name="naive_previous_day",
            target_column=prepared.target_column,
            start_offset=prepared.sequence_length,
        ),
        evaluate_lag_baseline(
            test_frame,
            lag_hours=24 * 7,
            model_name="seasonal_naive_previous_week",
            target_column=prepared.target_column,
            start_offset=prepared.sequence_length,
        ),
    ]
    results.extend(
        evaluate_tabular_baselines(
            prepared.frames["train"],
            prepared.frames["test"],
            target_column=prepared.target_column,
            feature_columns=prepared.feature_columns,
            include_xgboost=not args.skip_xgboost,
        )
    )

    model_path = Path(args.model_path)
    if model_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_path, device)
        scaled_predictions = predict_lstm(
            model,
            prepared.X_test,
            batch_size=args.batch_size,
            device=device,
        ).numpy()
        y_pred = prepared.target_scaler.inverse_transform(scaled_predictions).reshape(-1)
        y_true = prepared.target_scaler.inverse_transform(prepared.y_test).reshape(-1)
        results.append(regression_metrics(y_true, y_pred, "lstm"))
        save_forecast_plot(y_true, y_pred, Path(args.figure_path), args.plot_points)
    else:
        print(f"Model file not found at {model_path}; reporting baselines only.")

    metrics = pd.DataFrame([result.__dict__ for result in results]).sort_values("mae")
    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
