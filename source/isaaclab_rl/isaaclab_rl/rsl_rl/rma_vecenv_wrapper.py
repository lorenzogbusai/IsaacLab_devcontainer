# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import gymnasium as gym
from typing import Callable, Any

from .vecenv_wrapper import RslRlVecEnvWrapper


class RMAVecEnvWrapper(RslRlVecEnvWrapper):
    """RMA-specific wrapper for teacher-student training.

    This wrapper modifies the environment to support RMA (Rapid Motor Adaptation)
    training where a student policy learns to imitate a teacher policy.
    """

    def __init__(
        self,
        env: gym.Env,
        teacher_policy: Callable[[Any], torch.Tensor] | None = None,
        training_stage: str = "teacher"
    ):
        """Initialize the RMA wrapper.

        Args:
            env: The environment to wrap
            teacher_policy: Callable that takes environment observation and returns teacher actions
            training_stage: Either "teacher" or "student"
        """
        super().__init__(env)
        self.teacher_policy = teacher_policy
        self.training_stage = training_stage

        if training_stage == "student" and teacher_policy is None:
            raise ValueError("teacher_policy must be provided for student training stage")

    def step(self, actions):
        """Step the environment with actions."""
        # Call parent step method to get properly formatted results
        obs, rewards, dones, extras = super().step(actions)
        
        if self.training_stage == "student" and self.teacher_policy is not None:
            # Get teacher actions for imitation learning
            teacher_actions = self.teacher_policy(self.env.unwrapped)
            # Could add imitation loss here or modify rewards
            # For now, just store teacher actions in extras for potential use
            extras['teacher_actions'] = teacher_actions

        return obs, rewards, dones, extras

    def reset(self):
        """Reset the environment."""
        return self.env.reset()