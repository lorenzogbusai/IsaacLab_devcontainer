import torch
import torch.nn as nn
from typing import Tuple


def _make_mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, activation: str = "relu") -> nn.Sequential:
    layers = []
    prev = in_dim
    act = nn.ReLU if activation == "relu" else nn.Tanh
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(act())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class RMAEncoder(nn.Module):
    """Simple encoder used by RMA. Supports MLP and single-layer LSTM modes.

    Args:
        in_dim: dimensionality of flattened context input
        latent_dim: output latent dimension
        hidden: hidden sizes for MLP
        encoder_type: "mlp" or "rnn"
    """

    def __init__(self, in_dim: int, latent_dim: int, hidden: Tuple[int, ...] = (128, 128), encoder_type: str = "mlp"):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "mlp":
            self.net = _make_mlp(in_dim, hidden, latent_dim)
        elif encoder_type == "rnn":
            # use a single-layer LSTM and project hidden -> latent
            self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden[0], batch_first=True)
            self.proj = nn.Linear(hidden[0], latent_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Expects x shaped either (N, D) for MLP or (N, T, D) for RNN.
        Returns (N, latent_dim).
        """
        if self.encoder_type == "mlp":
            # flatten last dims if present
            if x.dim() > 2:
                x = x.reshape(x.shape[0], -1)
            return self.net(x)
        else:
            # rnn expects (N, T, D)
            out, (h, c) = self.rnn(x)
            # use last hidden state
            return self.proj(h[-1])
