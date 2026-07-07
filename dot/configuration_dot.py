#!/usr/bin/env python

"""Configuration for the DOT (Decoder-Only Transformer) policy.

Adapted from: https://github.com/IliaLarchenko/dot_policy
Ported to LeRobot's PreTrainedConfig system by Ryan.
"""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("dot")
@dataclass
class DOTConfig(PreTrainedConfig):
    """Configuration for DOT (Decoder-Only Transformer) policy.

    The DOT policy is a decoder-only transformer architecture for imitation learning
    that uses LoRA-adapted vision backbones and a causal transformer decoder for
    action prediction.

    Key parameters to adjust:
    - train_horizon / inference_horizon: number of action steps to predict
    - alpha / train_alpha: exponential weighting factors for action chunking
    - predict_every_n / return_every_n: inference speed optimization
    - n_obs_steps / lookback_obs_steps / lookback_aug: observation history parameters
    """

    # Input / output structure.
    # Tuned for a real SO-101 pick-and-place task at 30 fps
    # (2 cameras: wrist + top, 640x480).
    n_obs_steps: int = 3
    train_horizon: int = 75
    inference_horizon: int = 50
    lookback_obs_steps: int = 30
    lookback_aug: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Optional dataset stats override (useful for environments like PushT).
    override_dataset_stats: bool = False
    new_dataset_stats: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            "action": {"max": [512.0] * 2, "min": [0.0] * 2},
            "observation.environment_state": {"max": [512.0] * 16, "min": [0.0] * 16},
            "observation.state": {"max": [512.0] * 2, "min": [0.0] * 2},
        }
    )

    # Architecture.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    pre_norm: bool = True
    lora_rank: int = 20
    merge_lora: bool = False

    dim_model: int = 128
    n_heads: int = 8
    dim_feedforward: int = 512
    n_decoder_layers: int = 8
    rescale_shape: tuple[int, int] = (240, 320)

    # Augmentation.
    crop_scale: float = 0.8
    state_noise: float = 0.01
    noise_decay: float = 0.999995

    # Training and loss computation.
    dropout: float = 0.1

    # Weighting and inference.
    alpha: float = 0.98
    train_alpha: float = 0.99
    predict_every_n: int = 1
    return_every_n: int = 1

    # Training preset
    optimizer_lr: float = 1.0e-4
    optimizer_weight_decay: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.predict_every_n > self.inference_horizon:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be less than or equal to "
                f"horizon ({self.inference_horizon})."
            )
        if self.return_every_n > self.inference_horizon:
            raise ValueError(
                f"return_every_n ({self.return_every_n}) must be less than or equal to "
                f"horizon ({self.inference_horizon})."
            )
        if self.predict_every_n > self.inference_horizon // self.return_every_n:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be less than or equal to "
                f"horizon // return_every_n ({self.inference_horizon // self.return_every_n})."
            )
        if self.train_horizon < self.inference_horizon:
            raise ValueError(
                f"train_horizon ({self.train_horizon}) must be greater than or equal to "
                f"horizon ({self.inference_horizon})."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )
        if not self.action_feature:
            raise ValueError(
                "No action feature found in output_features. "
                "Make sure to use make_policy() or set output_features with an ACTION feature "
                "before instantiating DOTPolicy."
            )

    @property
    def observation_delta_indices(self) -> list:
        far_past_obs = list(
            range(
                -self.lookback_aug - self.lookback_obs_steps,
                self.lookback_aug + 1 - self.lookback_obs_steps,
            )
        )
        recent_obs = list(range(2 - self.n_obs_steps, 1))
        return far_past_obs + recent_obs

    @property
    def action_delta_indices(self) -> list:
        far_past_actions = list(
            range(
                -self.lookback_aug - self.lookback_obs_steps,
                self.lookback_aug + 1 - self.lookback_obs_steps,
            )
        )
        recent_actions = list(range(2 - self.n_obs_steps, self.train_horizon))
        return far_past_actions + recent_actions

    @property
    def reward_delta_indices(self) -> None:
        return None
