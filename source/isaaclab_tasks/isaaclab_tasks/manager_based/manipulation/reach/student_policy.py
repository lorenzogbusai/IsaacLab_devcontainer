# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from typing import Any


class StudentResidualPolicy(nn.Module):
    """Student policy for RMA that learns residuals on top of teacher actions.

    The student takes regular observations + teacher actions and outputs residuals
    that are added to the teacher actions to get the final actions.
    """

    def __init__(self, obs_dim: int, action_dim: int, teacher_action_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.teacher_action_dim = teacher_action_dim

        # Input is observations + teacher actions
        input_dim = obs_dim + teacher_action_dim

        # Residual policy network
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))  # Output residuals
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, teacher_actions: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual policy network."""
        # Concatenate observations and teacher actions
        combined_input = torch.cat([obs, teacher_actions], dim=-1)
        residuals = self.net(combined_input)
        return residuals

    def __call__(self, env, teacher_policy) -> torch.Tensor:
        """Get residual actions for RMA student.

        Args:
            env: The environment
            teacher_policy: The trained teacher policy

        Returns:
            Final actions (teacher_actions + residuals)
        """
        # Get policy observations
        policy_obs = env.observation_manager.compute_group("policy")

        # Get teacher actions
        with torch.no_grad():
            teacher_actions = teacher_policy(env)

        # Get residuals from student policy
        with torch.no_grad():
            residuals = self.forward(policy_obs, teacher_actions)

        # Final actions = teacher actions + residuals
        final_actions = teacher_actions + residuals

        return final_actions