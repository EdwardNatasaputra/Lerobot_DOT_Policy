#!/usr/bin/env python

"""Processor pipeline for the DOT (Decoder-Only Transformer) policy.

Creates the pre- and post-processing pipelines for normalization,
batching, device placement, and unnormalization.
"""

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_dot import DOTConfig


def make_dot_pre_post_processors(
    config: DOTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the DOT policy.

    The pre-processing pipeline handles normalization, batching, and device placement.
    The post-processing pipeline handles unnormalization and moves outputs to CPU.

    Args:
        config: The DOT policy configuration object.
        dataset_stats: Dataset statistics (e.g., mean, std, min, max) for normalization.

    Returns:
        A tuple of (pre-processor pipeline, post-processor pipeline).
    """
    # Override dataset stats if configured
    if config.override_dataset_stats and dataset_stats is not None:
        for k, v in config.new_dataset_stats.items():
            if k not in dataset_stats:
                dataset_stats[k] = {}
            for k1, v1 in v.items():
                dataset_stats[k][k1] = torch.tensor(v1)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
