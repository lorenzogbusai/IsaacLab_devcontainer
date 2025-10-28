# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlDppoAlgorithmCfg, RslRlSymmetryCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import anymal


@configclass
class AnymalDRoughDppoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_d_rough_dppo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDppoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # DPPO-specific defaults
        critic_network="qrdqn",
        qrdqn_quantile_count=200,
        value_loss="quantile_l1",
        value_lambda=0.95,
    )


@configclass
class AnymalDFlatDppoRunnerCfg(AnymalDRoughDppoRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat_dppo"
        self.policy.actor_hidden_dims = [128, 128, 128]