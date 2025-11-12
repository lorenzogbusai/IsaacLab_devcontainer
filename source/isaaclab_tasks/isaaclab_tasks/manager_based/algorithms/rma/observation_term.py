import weakref
from typing import Sequence

import torch
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.envs.utils.io_descriptors import GenericObservationIODescriptor
from isaaclab.utils.buffers.circular_buffer import CircularBuffer
from isaaclab.envs import ManagerBasedEnv

from .config import RmaObservationCfg
from .encoder import RMAEncoder
from .adapter import RMAAdapter


class RmaObservationTerm(ManagerTermBase):
    """Observation term that provides a per-env latent for Rapid Motor Adaptation.

    This is a lightweight prototype implementation. It maintains a circular buffer of a
    user-specified context vector and exposes a latent computed by an encoder + adapter.
    It supports a simple per-env inner-loop adaptation mechanism.
    """

    def __init__(self, cfg: RmaObservationCfg, env):
        super().__init__(cfg, env)
        # configuration
        self.cfg: RmaObservationCfg = cfg
        self._env = env

        # context dimension expected (user can provide via params)
        self.history_length = int(self.cfg.history_length)

        # buffers for state, actions, rewards
        self._state_buffer = CircularBuffer(max_len=self.history_length, batch_size=self.num_envs, device=self.device)
        self._action_buffer = CircularBuffer(max_len=self.history_length, batch_size=self.num_envs, device=self.device)
        self._reward_buffer = CircularBuffer(max_len=self.history_length, batch_size=self.num_envs, device=self.device)

        # placeholder: will determine context dim from sim signals on first call
        self._context_proj = None

        # encoder will be created lazily once we know the flattened context dim
        self.encoder = None

        # adapter: created lazily after encoder once we know action/state dims
        self.adapter = None

        # descriptor used by observation manager when inspecting
        self._descriptor = GenericObservationIODescriptor(name=self.cfg.name, shape=(self.cfg.latent_dim,), dtype=str(torch.float32))

        # internal step counter for scheduling updates
        self._step = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # clear context buffers and adapter fast-weights
        self._context_buffer.reset(batch_ids=env_ids)
        if self.adapter.fast_W is not None:
            # reset fast deltas for specified envs
            if env_ids is None:
                self.adapter.fast_W.zero_()
                self.adapter.fast_b.zero_()
            else:
                # env_ids is a sequence; map to tensor slice if needed
                indices = env_ids
                self.adapter.fast_W[indices] = 0.0
                self.adapter.fast_b[indices] = 0.0

    def __call__(self, env, asset_name=None, joint_ids=None, ee_body_name=None, context_dim=None, action_dim=None, state_dim=None, teacher_policy=None, inspect: bool = False):
        """Compute and return the RMA latent for all envs. If inspect=True, the manager will
        read the IO descriptor stored at `self._descriptor`.
        """
        if inspect:
            return torch.zeros((self.num_envs, self.cfg.latent_dim))

        # Build and append real sim signals into buffers
            asset_name = self.cfg.params.get("asset_name", "robot")
            asset = self._env.scene[asset_name]

            # joint selection
            joint_ids = self.cfg.params.get("joint_ids", slice(None))
            try:
                q = asset.data.joint_pos[:, joint_ids]
                qdot = asset.data.joint_vel[:, joint_ids]
                tau = asset.data.applied_torque[:, joint_ids]
            except Exception:
                # fallback to zeros if any field missing
                n_j = asset.num_total_dofs if hasattr(asset, "num_total_dofs") else 0
                q = torch.zeros((self.num_envs, n_j), device=self.device)
                qdot = torch.zeros_like(q)
                tau = torch.zeros_like(q)

            # end-effector position if requested
            ee_vec = []
            ee_name = self.cfg.params.get("ee_body_name", None)
            if ee_name is not None:
                try:
                    body_idx = asset.body_names.index(ee_name)
                    ee_pos = asset.data.body_pose_w[:, body_idx, :3]
                    ee_vec = [ee_pos]
                except Exception:
                    ee_vec = [torch.zeros((self.num_envs, 3), device=self.device)]
            # last action from action manager
            try:
                last_action = self._env.action_manager.prev_action.to(self.device)
            except Exception:
                last_action = torch.zeros((self.num_envs, 0), device=self.device)

            # last reward attempt
            last_reward = None
            if hasattr(self._env, "extras") and isinstance(self._env.extras, dict):
                last_reward = self._env.extras.get("last_reward", None)
            if last_reward is None:
                # try common RL buffer name
                last_reward = self._env.extras.get("r", torch.zeros(self.num_envs, device=self.device)) if isinstance(self._env.extras, dict) else torch.zeros(self.num_envs, device=self.device)
            if isinstance(last_reward, (list, tuple)):
                last_reward = torch.tensor(last_reward, device=self.device)
            if last_reward is None:
                last_reward = torch.zeros(self.num_envs, device=self.device)

            # assemble state vector per env
            state_parts = [q, qdot, tau] + ee_vec
            state = torch.cat([p.reshape(self.num_envs, -1) for p in state_parts], dim=-1)

            # append to buffers
            self._state_buffer.append(state)
            self._action_buffer.append(last_action)
            self._reward_buffer.append(last_reward.unsqueeze(-1))

            # prepare flattened context for encoder
            state_hist = self._state_buffer.buffer  # (N, H, state_dim)
            action_hist = self._action_buffer.buffer if self._action_buffer._buffer is not None else torch.zeros((self.num_envs, self.history_length, 0), device=self.device)
            reward_hist = self._reward_buffer.buffer

            context = torch.cat([state_hist.reshape(self.num_envs, -1), action_hist.reshape(self.num_envs, -1), reward_hist.reshape(self.num_envs, -1)], dim=-1)

            # lazy initialize encoder / adapter if needed
            if self.encoder is None:
                enc_in_dim = context.shape[1]
                # create a small projection if user supplied context_dim mismatch
                target_ctx_dim = int(self.cfg.params.get("context_dim", enc_in_dim))
                if target_ctx_dim != enc_in_dim:
                    # project to target dim
                    self._context_proj = torch.nn.Linear(enc_in_dim, target_ctx_dim).to(self.device)
                    enc_in_dim = target_ctx_dim
                    context = self._context_proj(context)
                self.encoder = RMAEncoder(in_dim=enc_in_dim if self.cfg.encoder_type == "mlp" else context.view(self.num_envs, self.history_length, -1).shape[-1], latent_dim=self.cfg.latent_dim, hidden=self.cfg.encoder_hidden, encoder_type=self.cfg.encoder_type)
                # create adapter with knowledge of action dim and state dim
                action_dim = None
                try:
                    action_dim = int(self._env.action_manager.action.shape[1])
                except Exception:
                    action_dim = self.cfg.params.get("action_dim", None)
                state_dim = state.shape[1]
                self.adapter = RMAAdapter(enc_dim=self.cfg.latent_dim if self.cfg.encoder_type == "rnn" else self.cfg.latent_dim, latent_dim=self.cfg.latent_dim, hidden=self.cfg.adapter_hidden, per_env_opt=self.cfg.per_env_opt, num_envs=self.num_envs, device=self.device, action_dim=action_dim, state_dim=state_dim)

            # prepare encoder input
            if self.cfg.encoder_type == "mlp":
                enc_in = context
            else:
                enc_in = context.view(self.num_envs, self.history_length, -1)

            enc_out = self.encoder(enc_in)

            # optionally run inner-loop adaptation using teacher imitation loss or next-state prediction
            diagnostics = {}
            if self.cfg.adapter_mode == "online" and self.cfg.inner_loop_steps > 0 and (self._step % max(1, self.cfg.update_frequency) == 0):
                # check for teacher policy in params
                teacher = self.cfg.params.get("teacher_policy", None)
                student = self.cfg.params.get("student_policy", None)
                if teacher is not None:
                    try:
                        # teacher is expected to be callable(env) -> (num_envs, action_dim)
                        teacher_actions = teacher(self._env)
                        # if a student policy callable is provided, pass it so adapt_inner_loop can compute exact student imitation loss
                        if student is not None:
                            diagnostics = self.adapter.adapt_inner_loop(
                                enc_out,
                                targets=teacher_actions,
                                target_type="action",
                                steps=self.cfg.inner_loop_steps,
                                lr=self.cfg.inner_loop_lr,
                                env_ids=None,
                                l2_reg=self.cfg.params.get("fast_delta_l2", 0.0),
                                student_policy=student,
                            )
                        else:
                            diagnostics = self.adapter.adapt_inner_loop(enc_out, targets=teacher_actions, target_type="action", steps=self.cfg.inner_loop_steps, lr=self.cfg.inner_loop_lr, env_ids=None, l2_reg=self.cfg.params.get("fast_delta_l2", 0.0))
                    except Exception:
                        diagnostics = {}
                else:
                    # next-state prediction target: predict current state from past context (exclude most recent)
                    try:
                        # use context excluding last entry for prediction
                        if self.history_length >= 2:
                            # build enc_out_pred from history excluding last frame
                            pred_hist = torch.cat([self._state_buffer.buffer[:, :-1, :].reshape(self.num_envs, -1), self._action_buffer.buffer[:, :-1, :].reshape(self.num_envs, -1), self._reward_buffer.buffer[:, :-1, :].reshape(self.num_envs, -1)], dim=-1)
                            if self._context_proj is not None:
                                pred_hist = self._context_proj(pred_hist)
                            if self.cfg.encoder_type == "mlp":
                                enc_pred_in = pred_hist
                            else:
                                enc_pred_in = pred_hist.view(self.num_envs, self.history_length - 1, -1)
                            enc_pred = self.encoder(enc_pred_in)
                            # target is current state (state_hist[:, -1, :])
                            target_state = state_hist[:, -1, :]
                            diagnostics = self.adapter.adapt_inner_loop(enc_pred, targets=target_state, target_type="state", steps=self.cfg.inner_loop_steps, lr=self.cfg.inner_loop_lr, env_ids=None, l2_reg=self.cfg.params.get("fast_delta_l2", 0.0))
                    except Exception:
                        diagnostics = {}

            self._step += 1

            # produce latent (conditioning) to be appended to policy observations
            latent = self.adapter(enc_out)

            # log diagnostics in env.extras for visibility
            if diagnostics:
                self._env.extras.setdefault("rma", {})["diagnostics"] = diagnostics

            return latent

    def state_dict(self):
        return {"encoder": self.encoder.state_dict(), "adapter": self.adapter.state_dict()}

    def load_state_dict(self, sd: dict):
        if "encoder" in sd:
            self.encoder.load_state_dict(sd["encoder"])
        if "adapter" in sd:
            self.adapter.load_state_dict(sd["adapter"])


def rma_observation_term(env: ManagerBasedEnv, asset_name: str = "robot", joint_ids: slice = slice(None), 
                         ee_body_name: str = None, context_dim: int = None, action_dim: int = None, 
                         state_dim: int = None, teacher_policy: callable = None, inspect: bool = False) -> torch.Tensor:
    """RMA observation term function that maintains state in the environment.
    
    This function creates and maintains RMA state in the environment's extras.
    """
    # Get or create RMA state
    if not hasattr(env, '_rma_state') or env._rma_state is None:
        env._rma_state = RmaObservationTerm(
            cfg=RmaObservationCfg(
                history_length=8,
                latent_dim=16,
                encoder_type="mlp",
                adapter_mode="online",
                inner_loop_steps=3,
                params={
                    "asset_name": asset_name,
                    "joint_ids": joint_ids,
                    "ee_body_name": ee_body_name,
                    "context_dim": context_dim,
                    "action_dim": action_dim,
                    "state_dim": state_dim,
                    "teacher_policy": teacher_policy,
                }
            ),
            env=env
        )
    
    result = env._rma_state(env, inspect=inspect)
    if result is None:
        # Fallback
        result = torch.zeros((env.num_envs, 16), device=env.device)
    return result
