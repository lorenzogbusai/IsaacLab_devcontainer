import torch
import torch.nn as nn
from typing import Optional, Sequence, Callable


class RMAAdapter(nn.Module):
    """Adapter network with optional per-env fast-weights for inner-loop adaptation.

    For simplicity the adapter is a single linear layer (optionally with hidden MLP),
    and per-env fast deltas are stored and updated during inner-loop steps.
    """

    def __init__(
        self,
        enc_dim: int,
        latent_dim: int,
        hidden: Sequence[int] | None = None,
        per_env_opt: bool = True,
        num_envs: int = 1,
        device: str = "cpu",
        action_dim: int | None = None,
        state_dim: int | None = None,
    ):
        super().__init__()
        self.device = device
        self.base = nn.Linear(enc_dim, latent_dim)
        self.per_env_opt = per_env_opt
        self.num_envs = num_envs

                # simple MLP option
        if hidden:
            layers = []
            prev = enc_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, latent_dim))
            self.base = nn.Sequential(*layers)

        # action / state prediction heads (used to compute supervised targets)
        self.action_dim = action_dim
        self.state_dim = state_dim
        if action_dim is not None:
            self.action_head = nn.Linear(latent_dim, action_dim)
        else:
            self.action_head = None
        if state_dim is not None:
            self.state_head = nn.Linear(latent_dim, state_dim)
        else:
            self.state_head = None

        # per-env fast deltas (initialized to zeros)
        if self.per_env_opt:
            # only support single linear base in delta mode for simplicity
            # we store deltas for weight and bias when base is Linear
            if isinstance(self.base, nn.Linear):
                W = self.base.weight.detach()
                b = self.base.bias.detach() if self.base.bias is not None else torch.zeros(W.shape[0], device=device)
                self.fast_W = torch.zeros((num_envs, *W.shape), device=device)
                self.fast_b = torch.zeros((num_envs, b.shape[0]), device=device)
            else:
                # for MLP, we store no per-env deltas (fallback)
                self.fast_W = None
                self.fast_b = None
        else:
            self.fast_W = None
            self.fast_b = None
