"""Build a SimpleNamespace from SwiftDetr Pydantic configs.

The namespace bridges Pydantic configs and the legacy builder functions
(build_model, build_backbone, build_criterion_and_postprocessors).
"""

from __future__ import annotations

import dataclasses
import types

from swiftdetr.config import ModelConfig, TrainConfig
from swiftdetr.models._defaults import MODEL_DEFAULTS, ModelDefaults

# ModelConfig fields forwarded to the namespace verbatim
_MC_NAMESPACE_FIELDS = {
    "amp",
    "backbone_lora",
    "bbox_reparam",
    "ca_nheads",
    "cls_loss_coef",
    "dec_layers",
    "dec_n_points",
    "device",
    "drop_path",
    "encoder",
    "encoder_imagenet_weights",
    "freeze_encoder",
    "gradient_checkpointing",
    "group_detr",
    "hidden_dim",
    "ia_bce_loss",
    "layer_norm",
    "lite_refpoint_refine",
    "mask_downsample_ratio",
    "num_channels",
    "num_classes",
    "num_queries",
    "num_select",
    "num_windows",
    "patch_size",
    "positional_encoding_size",
    "pretrain_weights",
    "projector_scale",
    "resolution",
    "sa_nheads",
    "segmentation_head",
    "two_stage",
    "use_rope",
}

# TrainConfig fields NOT forwarded to the namespace
_TC_NON_NAMESPACE_FIELDS = {
    "resume",
    "seed",
    "cls_loss_coef",
    "group_detr",
    "ia_bce_loss",
    "segmentation_head",
    "num_select",
    "accelerator",
    "strategy",
    "devices",
    "num_nodes",
    "tensorboard",
    "wandb",
    "mlflow",
    "project",
    "run",
    "auto_batch_target_effective",
    "auto_batch_max_targets_per_image",
    "auto_batch_ema_headroom",
    "progress_bar",
    "run_test",
    "dont_save_weights",
    "pin_memory",
    "persistent_workers",
    "lr_scheduler",
    "lr_min_factor",
    "class_names",
}


def _namespace_from_configs(
    model_config: ModelConfig,
    train_config: TrainConfig,
    defaults: ModelDefaults = MODEL_DEFAULTS,
) -> types.SimpleNamespace:
    """Build a SimpleNamespace from model + train configs and hardcoded defaults.

    Priority (highest to lowest): ModelConfig > TrainConfig > ModelDefaults.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
        defaults: Hardcoded architectural constants.

    Returns:
        A ``SimpleNamespace`` consumed by ``build_model`` and criterion builders.
    """
    ns: dict = {}

    # 1. Inject ModelDefaults (lowest priority)
    for field in dataclasses.fields(defaults):
        ns[field.name] = getattr(defaults, field.name)

    # 2. Forward TrainConfig fields (exclude non-namespace set)
    all_tc_fields = set(train_config.model_fields)
    for field_name in all_tc_fields - _TC_NON_NAMESPACE_FIELDS:
        ns[field_name] = getattr(train_config, field_name)

    # 3. Forward ModelConfig fields (highest priority, override TC)
    for field_name in _MC_NAMESPACE_FIELDS:
        if hasattr(model_config, field_name):
            ns[field_name] = getattr(model_config, field_name)

    # 4. Derived fields consumed by builder functions
    ns["num_feature_levels"] = len(model_config.projector_scale)
    # ``build_transformer`` reads ``decoder_norm``; map from ``ModelConfig.layer_norm``.
    if "layer_norm" in ns:
        ns["decoder_norm"] = "LN" if ns["layer_norm"] else "Identity"

    # 5. Transitional: ModelConfig.cls_loss_coef wins over TrainConfig
    if "cls_loss_coef" in model_config.model_fields_set:
        ns["cls_loss_coef"] = model_config.cls_loss_coef

    # 6. Segmentation loss fields (defaults when no segmentation head)
    if "mask_ce_loss_coef" not in ns:
        ns["mask_ce_loss_coef"] = 5.0
    if "mask_dice_loss_coef" not in ns:
        ns["mask_dice_loss_coef"] = 5.0
    if "mask_point_sample_ratio" not in ns:
        ns["mask_point_sample_ratio"] = 16

    return types.SimpleNamespace(**ns)
