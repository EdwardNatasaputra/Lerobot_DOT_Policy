#!/usr/bin/env python

"""The implementation of the Decoder-Only Transformer (DOT) policy.

Adapted from: https://github.com/IliaLarchenko/dot_policy
Ported to LeRobot's PreTrainedPolicy system by Ryan.

The DOT policy uses a decoder-only transformer with LoRA-adapted vision backbone
for efficient imitation learning. Key design choices:
- Shared backbone across cameras with per-camera projections
- Causal transformer decoder for action sequence prediction
- LoRA on Conv2d layers for parameter-efficient vision fine-tuning
- Exponentially-weighted action chunking for smooth inference
"""

import math

import torch
import torchvision
from torch import Tensor, nn
from torchvision import transforms
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import InterpolationMode

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_ENV_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_dot import DOTConfig


class DOTInnerModel(nn.Module):
    """Core DOT transformer model (decoder-only architecture).

    Processes observations through projection layers and a causal transformer
    decoder to predict action sequences.
    """

    def __init__(self, config: DOTConfig):
        super().__init__()
        self.config = config

        self.projections = nn.ModuleDict()
        self.n_features = 0

        self.image_names = sorted(config.image_features.keys())

        # Shared backbone for all cameras with LoRA adaptation
        if len(self.image_names) > 0:
            backbone = getattr(torchvision.models, self.config.vision_backbone)(
                weights=self.config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            backbone.fc = nn.Linear(backbone.fc.in_features, self.config.dim_model)
            self.projections["images"] = add_lora_to_backbone(backbone, rank=config.lora_rank)
            self.n_features += len(self.image_names) * self.config.n_obs_steps

        if self.config.robot_state_feature:
            self.projections["state"] = nn.Linear(
                self.config.robot_state_feature.shape[0], self.config.dim_model
            )
            self.n_features += self.config.n_obs_steps

        if self.config.env_state_feature:
            self.projections["environment_state"] = nn.Linear(
                self.config.env_state_feature.shape[0], self.config.dim_model
            )
            self.n_features += self.config.n_obs_steps

        self.projections_names = sorted(self.projections.keys())
        obs_mapping = {
            "images": OBS_IMAGES,
            "state": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
        }
        self.obs_mapping = {k: v for k, v in obs_mapping.items() if k in self.projections_names}

        # Extra trainable vector prepended to the input features
        self.prefix_input = nn.Parameter(torch.randn(1, 1, config.dim_model))

        # Setup transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.config.dim_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=self.config.pre_norm,
        )
        decoder_norm = nn.LayerNorm(self.config.dim_model)
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=self.config.n_decoder_layers, norm=decoder_norm
        )

        # Decoder uses non-trainable sinusoidal positional encodings
        decoder_pos = create_sinusoidal_pos_embedding(
            config.train_horizon + config.lookback_obs_steps, config.dim_model
        )
        decoder_pos = torch.cat(
            [
                decoder_pos[:1],
                decoder_pos[-config.train_horizon - config.n_obs_steps + 2 :],
            ],
            dim=0,
        )
        self.register_buffer("decoder_pos", decoder_pos)

        decoder_pos_inf = self.decoder_pos[
            : self.decoder_pos.shape[0] + self.config.inference_horizon - self.config.train_horizon
        ]
        self.register_buffer("decoder_pos_inf", decoder_pos_inf)

        mask = torch.zeros(len(decoder_pos), len(decoder_pos), dtype=torch.bool)
        mask[
            : len(decoder_pos) + config.inference_horizon - config.train_horizon,
            len(decoder_pos) + config.inference_horizon - config.train_horizon :,
        ] = True
        self.register_buffer("mask", mask)

        # Input features need trainable positional embeddings
        self.inputs_pos_emb = nn.Parameter(torch.empty(1, self.n_features, self.config.dim_model))
        nn.init.uniform_(
            self.inputs_pos_emb,
            -((1 / self.config.dim_model) ** 0.5),
            (1 / self.config.dim_model) ** 0.5,
        )

        # Final action regression head
        if self.config.action_feature is None:
            raise ValueError("No action feature found in output_features.")
        self.action_head = nn.Linear(self.config.dim_model, self.config.action_feature.shape[0])

    def _process_inputs(self, batch):
        """Project all inputs to the model dimension and concatenate them."""
        inputs_projections_list = []

        for state in self.projections_names:
            batch_state = self.obs_mapping[state]
            if batch_state in batch:
                bs, n_obs, *obs_shape = batch[batch_state].shape
                enc = self.projections[state](
                    batch[batch_state].view(bs * n_obs, *obs_shape)
                ).view(bs, n_obs, -1)
                inputs_projections_list.append(enc)

        return torch.cat(inputs_projections_list, dim=1)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        inputs_projections = self._process_inputs(batch)
        bs = inputs_projections.shape[0]

        inputs_projections += self.inputs_pos_emb.expand(bs, -1, -1)
        inputs_projections = torch.cat(
            [self.prefix_input.expand(bs, -1, -1), inputs_projections], dim=1
        )

        if self.training:
            decoder_out = self.decoder(
                self.decoder_pos.expand(bs, -1, -1), inputs_projections, self.mask
            )
        else:
            decoder_out = self.decoder(
                self.decoder_pos_inf.expand(bs, -1, -1), inputs_projections
            )
        return self.action_head(decoder_out)


class DOTPolicy(PreTrainedPolicy):
    """DOT (Decoder-Only Transformer) Policy for imitation learning.

    Uses the LeRobot PreTrainedPolicy interface with processor-based
    normalization/unnormalization (no internal Normalize/Unnormalize modules).
    """

    name: str = "dot"
    config_class: type[DOTConfig] = DOTConfig

    def __init__(
        self,
        config: DOTConfig,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.image_names = sorted(config.image_features.keys())

        self.model = DOTInnerModel(self.config)

        self.state_noise = self.config.state_noise
        self.crop_scale = self.config.crop_scale
        self.alpha = self.config.alpha
        self.inference_horizon = self.config.inference_horizon
        self.return_every_n = self.config.return_every_n
        self.predict_every_n = self.config.predict_every_n

        # Weights used for action chunking
        action_weights = self.alpha ** torch.arange(self.inference_horizon).float()
        action_weights /= action_weights.sum()
        action_weights = action_weights.view(1, -1, 1)
        self.register_buffer("action_weights", action_weights)

        # Weights for the loss computation (future actions weighted less)
        loss_weights = torch.ones(self.config.train_horizon + self.config.n_obs_steps - 1)
        loss_weights[-self.config.train_horizon :] = (
            self.config.train_alpha ** torch.arange(self.config.train_horizon).float()
        )
        loss_weights /= loss_weights.mean()
        loss_weights = loss_weights.view(1, -1, 1)
        self.register_buffer("loss_weights", loss_weights)

        # Image resize transform (nearest interpolation required for PushT-like envs)
        self.resize_transform = transforms.Resize(
            config.rescale_shape, interpolation=InterpolationMode.NEAREST
        )

        self.reset()

    def reset(self):
        """Reset inference state. Call whenever the environment resets."""
        self._old_predictions = None
        self._input_buffers = {}
        self._last_action = None
        self._step = 0

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def _update_observation_buffers(self, buffer_name: str, observation: Tensor) -> Tensor:
        """Maintain a rolling buffer of past observations for lookback.

        Keeps the last (lookback_obs_steps + 1) observations, returning the
        concatenation of the oldest and the most recent (n_obs_steps - 1).
        """
        if buffer_name not in self._input_buffers:
            self._input_buffers[buffer_name] = observation.unsqueeze(1).repeat(
                1,
                self.config.lookback_obs_steps + 1,
                *torch.ones(len(observation.shape[1:])).int(),
            )
        else:
            self._input_buffers[buffer_name] = self._input_buffers[buffer_name].roll(
                shifts=-1, dims=1
            )
            self._input_buffers[buffer_name][:, -1] = observation

        return torch.cat(
            [
                self._input_buffers[buffer_name][:, :1],
                self._input_buffers[buffer_name][:, -(self.config.n_obs_steps - 1) :],
            ],
            dim=1,
        )

    def _prepare_batch_for_inference(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare a batch for inference: resize images and update observation buffers."""
        # Resize and stack all images
        if len(self.image_names) > 0:
            batch[OBS_IMAGES] = torch.stack(
                [self.resize_transform(batch[k]) for k in self.image_names],
                dim=1,
            )  # bs, n_cam, c, h, w

        # Update observation queues for all inputs
        for name, batch_name in self.model.obs_mapping.items():
            if batch_name in batch:
                batch[batch_name] = self._update_observation_buffers(name, batch[batch_name])

        # Reshape images to keep the same order as during training
        if OBS_IMAGES in batch:
            batch[OBS_IMAGES] = batch[OBS_IMAGES].flatten(1, 2)
            # bs, n_obs * n_cam, c, h, w

        return batch

    def _chunk_actions(self, actions: Tensor) -> Tensor:
        """Exponentially-weighted action chunking for temporal smoothness."""
        if self._old_predictions is not None:
            self._old_predictions[:, 0] = actions
        else:
            self._old_predictions = actions.unsqueeze(1).repeat(
                1, self.config.inference_horizon, 1, 1
            )

        action = (self._old_predictions[:, :, 0] * self.action_weights).sum(dim=1)
        self._old_predictions = self._old_predictions.roll(shifts=(1, -1), dims=(1, 2))

        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Returns: (batch_size, inference_horizon, action_dim) tensor of predicted actions.
        """
        self.eval()
        batch = self._prepare_batch_for_inference(batch)
        actions_pred = self.model(batch)[:, -self.config.inference_horizon :]
        return actions_pred

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action for execution in the environment.

        Uses predict_every_n to skip redundant model calls, and return_every_n
        plus exponential chunking for smooth action selection.
        """
        self.eval()

        batch = self._prepare_batch_for_inference(batch)

        # Only run model prediction every predict_every_n steps
        if self._step % self.predict_every_n == 0 or self._last_action is None:
            actions_pred = self.model(batch)[:, -self.config.inference_horizon :]
            self._last_action = actions_pred
        else:
            # Shift previous predictions and repeat last action
            self._last_action = self._last_action.roll(-1, dims=1)
            self._last_action[:, -1] = self._last_action[:, -2]

        self._step += 1

        # Return chunked actions for return_every_n steps
        action = self._chunk_actions(self._last_action)
        for _ in range(self.return_every_n - 1):
            self._last_action = self._last_action.roll(-1, dims=1)
            self._last_action[:, -1] = self._last_action[:, -2]
            action = self._chunk_actions(self._last_action)

        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Applies data augmentation (random lookback, random crop, state noise)
        and computes weighted L1 loss with padding masking.
        """
        # Random lookback augmentation
        lookback_ind = torch.randint(0, 2 * self.config.lookback_aug + 1, (1,)).item()

        # Build the list of keys that need lookback slicing
        slice_keys = list(self.model.obs_mapping.values()) + list(self.image_names) + [
            "action", "action_is_pad"
        ]

        for k in slice_keys:
            if k != OBS_IMAGES and k in batch:
                batch[k] = torch.cat(
                    [
                        batch[k][:, lookback_ind : lookback_ind + 1],
                        batch[k][:, 2 * self.config.lookback_aug + 1 :],
                    ],
                    1,
                )

        if len(self.config.image_features) > 0:
            # Random crop augmentation
            scale = 1 - torch.rand(1) * (1 - self.crop_scale)
            new_shape = (
                int(self.config.rescale_shape[0] * scale),
                int(self.config.rescale_shape[1] * scale),
            )
            crop_transform = transforms.RandomCrop(new_shape)

            for k in self.image_names:
                bs, n_obs, c, h, w = batch[k].shape
                batch[k] = batch[k].view(bs * n_obs, c, h, w)
                batch[k] = crop_transform(self.resize_transform(batch[k]))
                batch[k] = batch[k].view(bs, n_obs, c, *batch[k].shape[-2:])

            batch[OBS_IMAGES] = torch.stack(
                [batch[k] for k in self.image_names], dim=2
            ).flatten(1, 2)
            # bs, n_obs * n_cam, c, h, w

        # Add random noise to states during training
        if self.state_noise is not None:
            for k in self.model.obs_mapping.values():
                if k != OBS_IMAGES and k in batch:
                    batch[k] += (torch.rand_like(batch[k]) * 2 - 1) * self.state_noise

        actions_hat = self.model(batch)

        loss = nn.functional.l1_loss(batch["action"], actions_hat, reduction="none")
        rev_padding = (~batch["action_is_pad"]).unsqueeze(-1)

        # Apply padding mask, loss weights, and decay
        loss = (loss * rev_padding * self.loss_weights).mean()

        loss_dict = {"l1_loss": loss.item()}

        # Reduce augmentation aggressiveness over training
        self.state_noise *= self.config.noise_decay
        self.crop_scale = 1 - (1 - self.crop_scale) * self.config.noise_decay

        return loss, loss_dict

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        """Load model from pretrained checkpoint and optionally merge LoRA."""
        policy = super().from_pretrained(pretrained_name_or_path, *args, **kwargs)

        if getattr(policy.config, "merge_lora", False):
            print("Merging LoRA after loading pretrained model...")
            policy.model = merge_lora_weights(policy.model)

        return policy


# ─── LoRA utilities ──────────────────────────────────────────────────────────


class LoRAConv2d(nn.Module):
    """Low-Rank Adaptation wrapper for Conv2d layers."""

    def __init__(self, base_conv, rank=4):
        super().__init__()
        self.base_conv = base_conv

        out_channels, in_channels, kh, kw = base_conv.weight.shape
        self.weight_shape = (out_channels, in_channels, kh, kw)
        fan_in = in_channels * kh * kw

        self.lora_A = nn.Parameter(torch.normal(0, 0.02, (out_channels, rank)))
        self.lora_B = nn.Parameter(torch.normal(0, 0.02, (rank, fan_in)))

    def forward(self, x):
        lora_update = torch.matmul(self.lora_A, self.lora_B).view(self.weight_shape)
        return nn.functional.conv2d(
            x,
            self.base_conv.weight + lora_update,
            self.base_conv.bias,
            stride=self.base_conv.stride,
            padding=self.base_conv.padding,
            dilation=self.base_conv.dilation,
            groups=self.base_conv.groups,
        )

    def merge_lora(self):
        """Merge LoRA weights into the base convolution and return a standard Conv2d layer."""
        lora_update = torch.matmul(self.lora_A, self.lora_B).view(self.weight_shape)
        self.base_conv.weight.copy_(self.base_conv.weight + lora_update)
        return self.base_conv


def replace_conv2d_with_lora(module, rank=4):
    """Recursively replace Conv2d layers with LoRAConv2d in the module."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank))
        else:
            replace_conv2d_with_lora(child, rank)
    return module


def merge_lora_weights(module):
    """Recursively merge LoRA weights in the module."""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAConv2d):
            setattr(module, name, child.merge_lora())
        else:
            merge_lora_weights(child)
    return module


def add_lora_to_backbone(backbone, rank=4, verbose=True):
    """Add LoRA adapters to a vision backbone, freezing all except LoRA and fc layers."""
    replace_conv2d_with_lora(backbone, rank)

    for name, param in backbone.named_parameters():
        if "lora_" in name or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return backbone


# ─── Positional embeddings ───────────────────────────────────────────────────


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """Create sinusoidal positional embeddings (Attention Is All You Need style)."""
    position = torch.arange(num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float) * (-math.log(10000.0) / dimension)
    )
    pe = torch.zeros(num_positions, dimension)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
