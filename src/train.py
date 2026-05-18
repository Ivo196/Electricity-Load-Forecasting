"""Train the LSTM forecasting model using a validation split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import SplitConfig, make_loaders, prepare_data
from .models import LSTMForecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/raw/continuous_dataset.csv")
    parser.add_argument("--model-path", default="models/lstm_forecast.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=24 * 7)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: LSTMForecast,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_rows = 0

    with torch.set_grad_enabled(is_training):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_training:
                optimizer.zero_grad()

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_rows += batch_size

    return total_loss / total_rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_loader, validation_loader, _ = make_loaders(prepared, batch_size=args.batch_size)

    model = LSTMForecast(
        input_size=len(prepared.feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_validation_loss = float("inf")
    best_state = None

    print(f"Device: {device}")
    for split_name, frame in prepared.frames.items():
        print(f"{split_name}: {frame.index.min()} -> {frame.index.max()} ({len(frame):,} rows)")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, criterion, device)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state or model.state_dict(),
            "input_size": len(prepared.feature_columns),
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "sequence_length": args.sequence_length,
            "feature_columns": prepared.feature_columns,
            "target_column": prepared.target_column,
            "split_config": {
                "train_size": args.train_size,
                "validation_size": args.validation_size,
                "test_size": args.test_size,
            },
        },
        model_path,
    )

    metadata_path = model_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "best_validation_loss": best_validation_loss,
                "model_path": str(model_path),
                "sequence_length": args.sequence_length,
                "feature_columns": prepared.feature_columns,
                "target_column": prepared.target_column,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved best model to {model_path}")


if __name__ == "__main__":
    main()
