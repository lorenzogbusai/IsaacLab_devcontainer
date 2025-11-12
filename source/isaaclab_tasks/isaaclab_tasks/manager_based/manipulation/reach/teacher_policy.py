# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from typing import Any


class SimpleTeacherPolicy(nn.Module):
    """Simple teacher policy for RMA that uses privileged observations.

    This is a basic policy that can serve as a teacher for RMA adaptation.
    For the reach task, it uses a simple proportional controller.
    """

    def __init__(self, action_dim: int, obs_dim: int = 35, hidden_dims: list[int] = [64, 64]):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        
        # Build network immediately with known dimensions
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network."""
        return self.net(obs)

    def __call__(self, env) -> torch.Tensor:
        """Get actions from privileged observations for RMA teacher."""
        # Get privileged observations
        privileged_obs = env.observation_manager.compute_group("privileged")
        # Flatten if needed
        if privileged_obs.dim() > 2:
            privileged_obs = privileged_obs.view(privileged_obs.shape[0], -1)

        with torch.no_grad():
            actions = self.forward(privileged_obs)

        return actions