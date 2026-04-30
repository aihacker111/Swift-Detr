# ------------------------------------------------------------------------
# SwiftDetr (detection config)
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os
from typing import List, Literal, Optional

import torch
from pydantic import BaseModel, field_validator

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class ModelConfig(BaseModel):
    encoder: Literal["swiftnet_tiny", "swiftnet_small", "swiftnet_base"]
    dec_layers: int
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5"]]
    hidden_dim: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    pretrained_encoder: Optional[str] = None  # Path to SwiftNet .pth weights
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    use_convnext_projector: bool = True  # True=ConvNeXt fusion, False=C2f (backbone projector)
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    freeze_encoder: bool = False  # Freeze SwiftNet backbone (no gradient update)
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    license: str = "Apache-2.0"

    @field_validator("pretrain_weights", "pretrained_encoder", mode="after")
    @classmethod
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        """
        Expand user paths (e.g., '~' or paths with separators) but leave simple filenames
        (like 'rf-detr-base.pth') unchanged so they can match hosted model keys.
        For pretrained_encoder "repo::weights" format, don't expand.
        """
        if v is None:
            return v
        if "::" in v:
            return v  # "repo::weights" format
        return os.path.realpath(os.path.expanduser(v))


class SwiftDetrTinyConfig(ModelConfig):
    """SwiftDetr-Tiny: SwiftNet-Tiny backbone (~6M encoder params)."""
    encoder: Literal["swiftnet_tiny", "swiftnet_small", "swiftnet_base"] = "swiftnet_tiny"
    hidden_dim: int = 256
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 8
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]
    pretrain_weights: Optional[str] = None
    resolution: int = 384


class SwiftDetrSmallConfig(ModelConfig):
    """SwiftDetr-Small: SwiftNet-Small backbone (~15M encoder params)."""
    encoder: Literal["swiftnet_tiny", "swiftnet_small", "swiftnet_base"] = "swiftnet_small"
    hidden_dim: int = 256
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 8
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]
    resolution: int = 512
    pretrain_weights: Optional[str] = None


class SwiftDetrBaseConfig(ModelConfig):
    """SwiftDetr-Base: SwiftNet-Base backbone at 640×640 (~30M encoder params)."""
    encoder: Literal["swiftnet_tiny", "swiftnet_small", "swiftnet_base"] = "swiftnet_base"
    hidden_dim: int = 256
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 8
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]
    pretrain_weights: Optional[str] = None
    resolution: int = 640


# --- Deprecated aliases (RF-DETR naming) ---
RFDETRNanoConfig = SwiftDetrTinyConfig
RFDETRSmallConfig = SwiftDetrSmallConfig
RFDETRBaseConfig = SwiftDetrBaseConfig
RFDETRLargeConfig = SwiftDetrBaseConfig

class TrainConfig(BaseModel):
    lr: float = 3e-4
    lr_encoder: float = 6e-5  # ~0.00017 sau sqrt(8), nhỏ hơn 8e-5
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 100
    resume: Optional[str] = None
    ema_decay: float = 0.993
    ema_tau: int = 100
    lr_drop: int = 100
    checkpoint_interval: int = 100
    warmup_epochs: float = 0.0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    group_detr: int = 13
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    use_varifocal_loss: bool = True  # Varifocal loss thay Focal loss cho classification
    use_convnext_projector: bool = True
    # Prototype Alignment (lwdetr_prototype) — EMA class prototypes cho query feature alignment
    use_prototype_align: bool = True
    prototype_loss_coef: float = 0.1
    prototype_momentum: float = 0.999      # EMA decay cho prototype update
    prototype_warmup_steps: int = 200      # Chỉ update prototype, chưa tính loss
    prototype_temperature: float = 0.1     # τ trong cosine classifier
    # Enhanced PrototypeMemory (lwdetr_prototype)
    prototype_repulsion_coef: float = 0.1  # [ENH-3] Inter-class repulsion loss weight
    prototype_use_freq_weight: bool = True # [ENH-2] Class-frequency weighting
    prototype_use_quality_weight: bool = True  # [ENH-4] Prototype quality weighting
    prototype_use_repulsion: bool = True   # [ENH-3] Toggle inter-class repulsion
    dataset_file: Literal["coco", "o365", "roboflow"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: str  # COCO layout root (train/, val/, annotations/); also default for coco_path when unset
    coco_path: Optional[str] = None  # If None, train_from_config uses dataset_dir for build_coco
    output_dir: str = "output"
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    use_ema: bool = True
    num_workers: int = 2
    weight_decay: float = 1e-4
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    tensorboard: bool = True
    wandb: bool = False
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: List[str] = None
    run_test: bool = False
    segmentation_head: bool = False
    eval_max_dets: int = 500
    freeze_encoder: bool = False  # Freeze SwiftNet backbone (no gradient update)

    @field_validator("dataset_dir", "output_dir", "coco_path", mode="after")
    @classmethod
    def expand_paths(cls, v: str | None) -> str | None:
        """
        Expand user paths (e.g., '~' or paths with separators) but leave simple filenames
        (like 'rf-detr-base.pth') unchanged so they can match hosted model keys.
        """
        if v is None:
            return v
        return os.path.realpath(os.path.expanduser(v))


class SegmentationTrainConfig(TrainConfig):
    mask_point_sample_ratio: int = 16
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    cls_loss_coef: float = 5.0
    segmentation_head: bool = True
