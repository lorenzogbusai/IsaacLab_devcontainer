from isaaclab.utils import configclass
from isaaclab.managers.manager_term_cfg import ObservationTermCfg


@configclass
class RmaObservationCfg(ObservationTermCfg):
    """Configuration for the Rapid Motor Adaptation observation term.

    This extends the standard ObservationTermCfg with RMA-specific hyperparameters.
    """

    # RMA hyper-parameters
    latent_dim: int = 16
    history_length: int = 8
    encoder_type: str = "mlp"  # "mlp" | "rnn"
    encoder_hidden: tuple[int, ...] = (128, 128)
    encoder_activation: str = "relu"

    adapter_mode: str = "stateless"  # "stateless" | "online"
    adapter_hidden: tuple[int, ...] = (64,)

    inner_loop_steps: int = 3
    inner_loop_lr: float = 1e-2
    update_frequency: int = 1
    warmup_steps: int = 1

    injection_method: str = "concat"  # currently only "concat" supported
    normalize_input: bool = True
    per_env_opt: bool = True

    device: str | None = None
    name: str = "rma_latent"
