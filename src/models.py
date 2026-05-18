"""Model definitions."""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecast(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(X)
        last_timestep = output[:, -1, :]
        return self.head(last_timestep)
