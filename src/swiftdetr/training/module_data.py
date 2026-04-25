# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Swift-DETR dataset and dataloader builder (pure PyTorch, no Lightning)."""

from typing import Any, List, Optional, Tuple

import torch
import torch.utils.data
from torch.utils.data import DataLoader

from swiftdetr._namespace import _namespace_from_configs
from swiftdetr.config import ModelConfig, TrainConfig
from swiftdetr.datasets import build_dataset
from swiftdetr.datasets.aug_config import AUG_CONFIG
from swiftdetr.util.logger import get_logger
from swiftdetr.util.tensors import make_collate_fn

logger = get_logger()

_MIN_TRAIN_BATCHES = 5


def _has_cuda_device() -> bool:
    from swiftdetr.config import DEVICE
    return str(DEVICE).startswith("cuda")


def _resolve_augmentation_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    if not _has_cuda_device():
        return "cpu"
    try:
        import kornia.augmentation  # noqa: F401
        return "gpu"
    except ImportError:
        return "cpu"


class GradAccumAlignedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that pads length to a multiple of ``effective_batch_size * world_size``.

    Workaround for Lightning issue #19987 (kept for correctness even in pure-PyTorch mode):
    ensures every accumulation window is complete so the optimizer always sees a full batch.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        effective_batch_size: int,
        world_size: int = 1,
    ) -> None:
        if effective_batch_size < 1:
            raise ValueError(f"effective_batch_size must be >= 1, got {effective_batch_size}")
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")

        self._dataset = dataset
        self._dataset_length = len(dataset)  # type: ignore[arg-type]
        pad_unit = effective_batch_size * world_size
        remainder = self._dataset_length % pad_unit
        pad_count = (pad_unit - remainder) % pad_unit
        pad_index_generator = torch.Generator()
        pad_index_generator.manual_seed(0)
        self._pad_indices: list[int] = (
            torch.randint(0, self._dataset_length, (pad_count,), generator=pad_index_generator).tolist()
            if pad_count > 0
            else []
        )
        self._length = self._dataset_length + pad_count

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Any:
        dataset_idx = idx if idx < self._dataset_length else self._pad_indices[idx - self._dataset_length]
        return self._dataset[dataset_idx]


class SwiftDetrData:
    """Builds datasets and dataloaders for Swift-DETR training (no Lightning)."""

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        self.model_config = model_config
        self.train_config = train_config

        block_size = model_config.patch_size * model_config.num_windows
        if block_size <= 0:
            raise ValueError(
                f"Computed collate block_size must be > 0, got {block_size} "
                f"from patch_size={model_config.patch_size} and num_windows={model_config.num_windows}."
            )
        self._collate_fn = make_collate_fn(block_size=block_size)

        from swiftdetr.config import DEVICE

        accelerator = str(self.train_config.accelerator).lower()
        uses_cuda = accelerator in {"auto", "gpu", "cuda"}
        self._pin_memory: bool = (
            (DEVICE == "cuda" and uses_cuda)
            if self.train_config.pin_memory is None
            else bool(self.train_config.pin_memory)
        )
        num_workers = self.train_config.num_workers
        self._num_workers = num_workers
        self._persistent_workers: bool = (
            num_workers > 0
            if self.train_config.persistent_workers is None
            else bool(self.train_config.persistent_workers)
        )
        self._prefetch_factor = (
            (self.train_config.prefetch_factor if self.train_config.prefetch_factor is not None else 2)
            if num_workers > 0
            else None
        )

        # Datasets (built lazily)
        self._dataset_train: Optional[torch.utils.data.Dataset] = None
        self._dataset_val: Optional[torch.utils.data.Dataset] = None

        # Kornia pipeline
        self._kornia_pipeline: Any = None
        self._kornia_normalize: Any = None

        self._setup()

    def _setup(self) -> None:
        resolution = self.model_config.resolution
        ns = _namespace_from_configs(self.model_config, self.train_config)

        resolved = _resolve_augmentation_backend(self.train_config.augmentation_backend)
        if resolved != self.train_config.augmentation_backend:
            ns.augmentation_backend = resolved

        self._dataset_train = build_dataset("train", ns, resolution)
        self._dataset_val = build_dataset("val", ns, resolution)
        self._setup_kornia(resolved)

    def _setup_kornia(self, backend: str) -> None:
        if backend == "cpu":
            return
        if backend == "auto":
            if not _has_cuda_device():
                return
            try:
                import kornia.augmentation  # noqa: F401
            except ImportError:
                return
        elif backend == "gpu":
            if not _has_cuda_device():
                raise RuntimeError("augmentation_backend='gpu' requires a CUDA device")
            try:
                import kornia.augmentation  # noqa: F401
            except ImportError as err:
                raise ImportError(
                    "GPU augmentation requires kornia. Install with: pip install 'swiftdetr[kornia]'"
                ) from err

        from swiftdetr.datasets.kornia_transforms import build_kornia_pipeline, build_normalize

        aug_config = self.train_config.aug_config if self.train_config.aug_config is not None else AUG_CONFIG
        self._kornia_pipeline = build_kornia_pipeline(aug_config, self.model_config.resolution)
        self._kornia_normalize = build_normalize()
        logger.info("Kornia GPU augmentation pipeline built (backend=%s)", backend)

    def train_dataloader(self, world_size: int = 1, rank: int = 0) -> DataLoader:
        dataset = self._dataset_train
        batch_size = self.train_config.batch_size
        effective_batch_size = batch_size * self.train_config.grad_accum_steps

        if len(dataset) < effective_batch_size * _MIN_TRAIN_BATCHES:
            logger.info(
                "Training with uniform sampler: dataset too small (%d < %d)",
                len(dataset),
                effective_batch_size * _MIN_TRAIN_BATCHES,
            )
            sampler = torch.utils.data.RandomSampler(
                dataset, replacement=True, num_samples=effective_batch_size * _MIN_TRAIN_BATCHES
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=self._collate_fn,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
                prefetch_factor=self._prefetch_factor,
            )

        dataset = GradAccumAlignedDataset(dataset, effective_batch_size, world_size)

        if world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=self._collate_fn,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
                prefetch_factor=self._prefetch_factor,
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset_val,
            batch_size=self.train_config.batch_size,
            sampler=torch.utils.data.SequentialSampler(self._dataset_val),
            drop_last=False,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    @property
    def cat_id_to_name(self) -> dict:
        for dataset in (self._dataset_train, self._dataset_val):
            if dataset is None:
                continue
            coco = getattr(dataset, "coco", None)
            if coco is not None and hasattr(coco, "cats"):
                if hasattr(coco, "label2cat"):
                    return {label: coco.cats[cat_id]["name"] for label, cat_id in coco.label2cat.items()}
                return {k: v["name"] for k, v in coco.cats.items()}
        return {}

    @property
    def class_names(self) -> Optional[List[str]]:
        for dataset in (self._dataset_train, self._dataset_val):
            if dataset is None:
                continue
            coco = getattr(dataset, "coco", None)
            if coco is not None and hasattr(coco, "cats"):
                return [coco.cats[k]["name"] for k in sorted(coco.cats.keys())]
        return None

    def transfer_batch_to_device(
        self, batch: Tuple, device: torch.device
    ) -> Tuple:
        """Move ``(NestedTensor, targets)`` to device. Called by the training loop."""
        samples, targets = batch
        non_blocking = device.type == "cuda"
        samples = samples.to(device, non_blocking=non_blocking)
        targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]
        return samples, targets
