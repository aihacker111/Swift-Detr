"""Swift-DETR configuration classes.

Three detection variants built on SwiftNet backbones:
    SwiftDetrTinyConfig   — swiftnet_tiny  (~6M  backbone)
    SwiftDetrSmallConfig  — swiftnet_small (~15M backbone)
    SwiftDetrBaseConfig   — swiftnet_base  (~30M backbone)

All variants share a 256-dim transformer and 300 queries.
Input must be divisible by 32 (SwiftNet stride constraint).
"""

from __future__ import annotations

import os
import warnings
from typing import Any, ClassVar, List, Literal, Mapping, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SwiftNetVariant = Literal["swiftnet_tiny", "swiftnet_small", "swiftnet_base"]


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = _detect_device()


class BaseConfig(BaseModel):
    """Strict Pydantic base — rejects unknown fields with an actionable error."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def catch_typo_kwargs(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        allowed = set(cls.model_fields)
        unknown = set(values) - allowed
        if unknown:
            unknown_list = ", ".join(f"'{p}'" for p in sorted(unknown))
            allowed_list = ", ".join(sorted(allowed))
            raise ValueError(
                f"Unknown parameter(s): {unknown_list}. "
                f"Available parameter(s): {allowed_list}."
            )
        return values

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in type(self).model_fields:
            super().__setattr__(name, value)
            return
        raise ValueError(f"Unknown attribute: '{name}'.")


class ModelConfig(BaseConfig):
    """Architecture configuration for Swift-DETR models.

    Note:
        ``patch_size=32`` encodes the input-divisibility requirement of the
        SwiftNet backbone (stem /4  × three 2× mergers = stride 32).
        ``num_windows=1`` is unused by SwiftNet but kept for shape-validation
        compatibility: ``block_size = patch_size * num_windows = 32``.
    """

    # ── Backbone ──────────────────────────────────────────────────────────────
    encoder: SwiftNetVariant
    drop_path: float = 0.0
    freeze_encoder: bool = False
    # Path to a SWIFTNet ImageNet-1K (or compatible trunk) checkpoint. Loaded into
    # the detection backbone before training / before ``pretrain_weights`` (if set).
    encoder_imagenet_weights: Optional[str] = None

    # ── Feature pyramid ───────────────────────────────────────────────────────
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]

    # ── Transformer ───────────────────────────────────────────────────────────
    hidden_dim: int = 256
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    two_stage: bool = True
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    # RoPE 2D decoder self-attention (requires hidden_dim divisible by 4).
    use_rope: bool = True

    # ── Queries ───────────────────────────────────────────────────────────────
    num_queries: int = 300
    num_select: int = 300
    group_detr: int = 13

    # ── Input ─────────────────────────────────────────────────────────────────
    resolution: int
    # SwiftNet requires inputs divisible by 32.
    # patch_size * num_windows = 32 satisfies the shape validator.
    patch_size: int = 32
    num_windows: int = 1
    num_channels: int = Field(default=3, ge=1)

    # ── Classes & loss ────────────────────────────────────────────────────────
    num_classes: int = 90
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4

    # ── Position encoding ─────────────────────────────────────────────────────
    # Default: resolution // 16 (P4 stride).  Auto-synced when resolution changes.
    positional_encoding_size: int

    # ── Misc ──────────────────────────────────────────────────────────────────
    amp: bool = True
    gradient_checkpointing: bool = False
    compile: bool = False
    fused_optimizer: bool = True
    backbone_lora: bool = False
    pretrain_weights: Optional[str] = None
    # When True and ``pretrain_weights`` is set, Lightning training will load that
    # checkpoint (and may download if the value is a hub/registry key). Default False
    # avoids any pretrained detection init for scratch training. Ignored when loading
    # for inference (``load_pretrain_weights(..., for_inference=True)``).
    load_detection_pretrain: bool = False
    device: str = DEVICE
    model_name: Optional[str] = Field(
        default=None,
        description="Stored in checkpoints to enable from_checkpoint() class resolution.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _sync_pe_with_resolution(self) -> "ModelConfig":
        """Keep positional_encoding_size = resolution // 16 when resolution changes."""
        if (
            "resolution" in self.model_fields_set
            and "positional_encoding_size" not in self.model_fields_set
        ):
            self.positional_encoding_size = self.resolution // 16
        return self

    @field_validator("pretrain_weights", "encoder_imagenet_weights", mode="after")
    @classmethod
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        expanded = os.path.expanduser(v)
        return os.path.realpath(expanded)

    @field_validator("device", mode="before")
    @classmethod
    def normalize_device(cls, v: Any) -> str:
        if isinstance(v, torch.device):
            return str(v)
        if isinstance(v, str):
            return str(torch.device(v))
        raise ValueError("device must be a string or torch.device.")


# ── Concrete model variant configs ─────────────────────────────────────────


class SwiftDetrTinyConfig(ModelConfig):
    """Swift-DETR Tiny — ~6M backbone, suited for edge / real-time inference."""

    encoder: SwiftNetVariant = "swiftnet_tiny"
    resolution: int = 512
    positional_encoding_size: int = 512 // 16  # 32
    dec_layers: int = 2
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]


class SwiftDetrSmallConfig(ModelConfig):
    """Swift-DETR Small — ~15M backbone, balanced speed / accuracy."""

    encoder: SwiftNetVariant = "swiftnet_small"
    resolution: int = 640
    positional_encoding_size: int = 640 // 16  # 40
    dec_layers: int = 3
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]


class SwiftDetrBaseConfig(ModelConfig):
    """Swift-DETR Base — ~30M backbone, high accuracy for server-class edge."""

    encoder: SwiftNetVariant = "swiftnet_base"
    resolution: int = 640
    positional_encoding_size: int = 640 // 16  # 40
    dec_layers: int = 4
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]


# ── Training config ────────────────────────────────────────────────────────


class TrainConfig(BaseModel):
    """Training hyperparameters for Swift-DETR."""

    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    batch_size: Union[int, Literal["auto"]] = 4
    grad_accum_steps: int = 4
    auto_batch_target_effective: int = 16
    auto_batch_max_targets_per_image: int = 100
    auto_batch_ema_headroom: float = 0.7
    epochs: int = 100
    resume: Optional[str] = None
    ema_decay: float = 0.993
    ema_tau: int = 100
    lr_drop: int = 100
    checkpoint_interval: int = Field(default=50, ge=1)
    warmup_epochs: float = 0.0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    group_detr: int = 13
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    num_select: int = 300
    dataset_file: Literal["coco", "roboflow", "yolo"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: str
    output_dir: str = "output"
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    use_ema: bool = True
    ema_update_interval: int = 1
    num_workers: int = 2
    weight_decay: float = 1e-4
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    progress_bar: Optional[Literal["tqdm", "rich"]] = None
    tensorboard: bool = True
    wandb: bool = False
    mlflow: bool = False
    clearml: bool = False
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: Optional[List[str]] = None
    run_test: bool = False
    segmentation_head: bool = False
    eval_max_dets: int = 500
    eval_interval: int = 1
    log_per_class_metrics: bool = True
    aug_config: Optional[Any] = None
    augmentation_backend: Literal["cpu", "auto", "gpu"] = "cpu"
    save_dataset_grids: bool = False
    accelerator: str = "auto"
    clip_max_norm: float = 0.1
    seed: Optional[int] = None
    sync_bn: bool = False
    strategy: str = "auto"
    devices: Union[int, str] = 1
    num_nodes: int = 1
    fp16_eval: bool = False
    lr_scheduler: Literal["step", "cosine"] = "step"
    lr_min_factor: float = 0.0
    dont_save_weights: bool = False
    train_log_sync_dist: bool = False
    train_log_on_step: bool = False
    compute_val_loss: bool = True
    compute_test_loss: bool = True
    pin_memory: Optional[bool] = None
    persistent_workers: Optional[bool] = None
    prefetch_factor: Optional[int] = None

    @field_validator("batch_size", mode="after")
    @classmethod
    def validate_batch_size(cls, v: Union[int, str]) -> Union[int, str]:
        if v == "auto":
            return v
        if v < 1:
            raise ValueError("batch_size must be >= 1, or 'auto'.")
        return v

    @field_validator("grad_accum_steps", "auto_batch_target_effective", mode="after")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Must be >= 1.")
        return v

    @field_validator("dataset_dir", "output_dir", mode="after")
    @classmethod
    def expand_paths(cls, v: str) -> str:
        return os.path.realpath(os.path.expanduser(v))

    @field_validator("progress_bar", mode="before")
    @classmethod
    def coerce_legacy_progress_bar(cls, value: Any) -> Any:
        if isinstance(value, bool):
            return "tqdm" if value else None
        return value
