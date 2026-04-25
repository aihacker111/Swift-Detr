"""SwiftDetr — public training and inference API."""

from __future__ import annotations

import operator
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

from swiftdetr.assets.coco_classes import COCO_CLASS_NAMES
from swiftdetr.config import ModelConfig, TrainConfig
from swiftdetr.datasets.coco import is_valid_coco_dataset
from swiftdetr.datasets.yolo import is_valid_yolo_dataset
from swiftdetr.inference import ModelContext, _build_model_context
from swiftdetr.util.distributed import is_main_process
from swiftdetr.util.logger import get_logger

if TYPE_CHECKING:
    import supervision as sv

torch.set_float32_matmul_precision("high")

logger = get_logger()

__all__ = ["SwiftDetr", "ModelContext"]

_VARIANT_EXPORTS = ("SwiftDetrTiny", "SwiftDetrSmall", "SwiftDetrBase")
_CHECKPOINT_MODEL_MAP: tuple[tuple[str, str], ...] = (
    ("swiftdetr-base",  "SwiftDetrBase"),
    ("swiftdetr-small", "SwiftDetrSmall"),
    ("swiftdetr-tiny",  "SwiftDetrTiny"),
)


def _validate_shape_dims(shape: object, block_size: int) -> tuple[int, int]:
    """Validate that (height, width) are divisible by block_size.

    Args:
        shape: Two-element sequence ``(height, width)``.
        block_size: Required divisor (32 for SwiftNet).

    Returns:
        ``(height, width)`` as plain ints.

    Raises:
        ValueError: On invalid shape or non-divisible dimensions.
    """
    try:
        height, width = shape  # type: ignore[misc]
    except (TypeError, ValueError):
        raise ValueError(f"shape must be (height, width), got {shape!r}.")
    for dim_name, dim in (("height", height), ("width", width)):
        if isinstance(dim, bool):
            raise ValueError(f"shape {dim_name} must be an integer, got {type(dim).__name__}.")
        operator.index(dim)  # raises TypeError for non-integer types
        if dim <= 0:
            raise ValueError(f"shape dimensions must be positive, got {shape!r}.")
    height, width = operator.index(height), operator.index(width)
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            f"shape dimensions must both be divisible by {block_size} "
            f"(SwiftNet stride), got {shape!r}."
        )
    return height, width


def _ensure_model_on_device(model_ctx: Any) -> None:
    """Move model weights to the target device recorded in model_ctx."""
    target = getattr(model_ctx, "device", None)
    inner = getattr(model_ctx, "model", None)
    if target is None or inner is None or not hasattr(inner, "parameters"):
        return
    if isinstance(target, str):
        target = torch.device(target)
    first_param = next(inner.parameters(), None)
    if first_param is not None and first_param.device != target:
        model_ctx.model = inner.to(target)


class SwiftDetr:
    """Base class for all Swift-DETR detection models.

    Provides training, inference, and checkpoint management.  Concrete
    model sizes are defined as subclasses in :mod:`swiftdetr.variants`.

    Args:
        **kwargs: Forwarded verbatim to the model config constructor
            (e.g. ``num_classes=10``, ``resolution=640``).
    """

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size: str | None = None
    _model_config_class: type[ModelConfig] = ModelConfig
    _train_config_class: type[TrainConfig] = TrainConfig

    def __init__(self, **kwargs: Any) -> None:
        self.model_config = self.get_model_config(**kwargs)
        self.model = self.get_model(self.model_config)
        self.callbacks: dict[str, list] = defaultdict(list)
        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None

    def get_model_config(self, **kwargs: Any) -> ModelConfig:
        """Instantiate the model config for this variant."""
        return self._model_config_class(**kwargs)

    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike[str], **kwargs: Any) -> SwiftDetr:
        """Load a Swift-DETR model from a training checkpoint.

        Automatically infers the model subclass from the ``model_name`` key
        written during training.  Falls back to ``pretrain_weights`` filename
        matching for older checkpoints.

        Args:
            path: Path to a checkpoint file (e.g. ``checkpoint_best_total.pth``).
            **kwargs: Additional arguments forwarded to the model constructor.

        Returns:
            An instance of the appropriate :class:`SwiftDetr` subclass.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            ValueError: If the model class cannot be inferred.
        """
        import swiftdetr.variants as variants_mod

        ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        args = ckpt["args"]

        # Build name → class mapping from the variants module
        _name_to_cls: dict[str, type[SwiftDetr]] = {
            getattr(obj, "__name__", sym): obj
            for sym in dir(variants_mod)
            if sym.startswith("SwiftDetr")
            for obj in [getattr(variants_mod, sym)]
            if isinstance(obj, type)
        }

        # Prefer model_name stored in checkpoint (written since v1.0)
        saved_name = ckpt.get("model_name", "")
        model_cls: type[SwiftDetr] | None = _name_to_cls.get(saved_name.strip()) if saved_name else None

        # Fallback: parse pretrain_weights filename
        if model_cls is None:
            weights_name = str(
                args.get("pretrain_weights", "") if isinstance(args, dict) else getattr(args, "pretrain_weights", "")
            ).lower()
            for name_fragment, class_name in _CHECKPOINT_MODEL_MAP:
                if name_fragment in weights_name:
                    model_cls = _name_to_cls.get(class_name)
                    break

        if model_cls is None:
            raise ValueError(
                f"Cannot infer model class from checkpoint {path!r}. "
                "Set model_name in the checkpoint or use the exact subclass directly."
            )

        instance = model_cls.__new__(model_cls)
        SwiftDetr.__init__(instance, pretrain_weights=str(path), **kwargs)
        return instance

    def get_model(self, model_config: ModelConfig) -> ModelContext:
        """Build the nn.Module and wrap it in a ModelContext.

        Args:
            model_config: Architecture configuration.

        Returns:
            :class:`ModelContext` with the model on CPU.
        """
        return _build_model_context(model_config)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, **kwargs: Any) -> None:
        """Train the model with PyTorch Lightning.

        Args:
            **kwargs: Forwarded to :class:`~swiftdetr.config.TrainConfig`
                (e.g. ``dataset_dir="/data/coco"``, ``epochs=100``).
        """
        from swiftdetr.training.trainer import build_trainer
        from swiftdetr.training.module_data import SwiftDetrDataModule
        from swiftdetr.training.module_model import SwiftDetrModule

        self.model_config.model_name = type(self).__name__
        train_config = self._train_config_class(**kwargs)

        data_module = SwiftDetrDataModule(self.model_config, train_config)
        model_module = SwiftDetrModule(
            model_config=self.model_config,
            train_config=train_config,
        )
        trainer = build_trainer(train_config)
        trainer.fit(model_module, datamodule=data_module)

    # ── Inference ─────────────────────────────────────────────────────────────

    @property
    def _block_size(self) -> int:
        return self.model_config.patch_size * self.model_config.num_windows  # = 32

    def predict(
        self,
        images: Any,
        threshold: float = 0.5,
        shape: tuple[int, int] | None = None,
    ) -> "sv.Detections":
        """Run object detection on one or more images.

        Args:
            images: A PIL Image, numpy array, file path (str/Path), URL string,
                or a list of any of the above.
            threshold: Minimum confidence score to retain a detection.
            shape: Optional ``(height, width)`` override for the inference
                resolution.  Both dimensions must be divisible by 32.

        Returns:
            :class:`supervision.Detections` with boxes, labels, and scores.
        """
        import supervision as sv

        _ensure_model_on_device(self.model)

        if shape is not None:
            h, w = _validate_shape_dims(shape, self._block_size)
            resolution = (h, w)
        else:
            resolution = self.model_config.resolution

        if not isinstance(images, list):
            images = [images]

        pil_images = [self._load_image(img) for img in images]

        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        preprocessed = torch.stack([
            self._preprocess(img, resolution) for img in pil_images
        ])

        model = self.model.model
        model.eval()
        device = self.model.device

        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=self.model_config.amp):
            preprocessed = preprocessed.to(device)
            outputs = model(preprocessed)

        original_sizes = torch.tensor(
            [[img.height, img.width] for img in pil_images],
            dtype=torch.float32,
            device=device,
        )
        results = self.model.postprocess(outputs, original_sizes)

        detections_list: list[sv.Detections] = []
        for result in results:
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            boxes = result["boxes"].cpu().numpy()

            keep = scores >= threshold
            detections_list.append(
                sv.Detections(
                    xyxy=boxes[keep],
                    confidence=scores[keep],
                    class_id=labels[keep].astype(int),
                )
            )

        return detections_list[0] if len(detections_list) == 1 else sv.Detections.merge(detections_list)

    def _load_image(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, (str, Path)):
            path = str(image)
            if path.startswith(("http://", "https://")):
                import requests
                from io import BytesIO
                response = requests.get(path, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            return Image.open(path).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def _preprocess(self, image: Image.Image, resolution: tuple[int, int]) -> torch.Tensor:
        image = image.resize((resolution[1], resolution[0]), Image.BILINEAR)
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, mean=self.means, std=self.stds)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on(self, event: str, callback: Any) -> None:
        """Register a callback for a training lifecycle event.

        Args:
            event: Event name (e.g. ``"on_fit_end"``).
            callback: Callable invoked when the event fires.
        """
        self.callbacks[event].append(callback)
