from isaaclab.utils import configclass

from .ik_rel_env_cfg import FrankaCabinetEnvCfg
from isaaclab_tasks.manager_based.algorithms.rma.config import RmaObservationCfg
from isaaclab_tasks.manager_based.algorithms.rma.observation_term import RmaObservationTerm


@configclass
class FrankaCabinetEnvCfg_RMA(FrankaCabinetEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # small debug-friendly settings
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.0

        # add RMA observation term to the policy observation group
        # we register the class to be instantiated by the ObservationManager
        self.observations.policy.rma = RmaObservationCfg(
            func=RmaObservationTerm,
            params={"context_dim": 32},
            latent_dim=16,
            history_length=4,
            encoder_type="mlp",
            adapter_mode="online",
            inner_loop_steps=3,
            inner_loop_lr=1e-2,
            per_env_opt=True,
        )
