"""ModelContext and model-context builder for Swift-DETR inference."""

from __future__ import annotations

__all__ = ["ModelContext"]

from typing import TYPE_CHECKING, Any, List, Optional

import torch

from swiftdetr.config import TrainConfig
from swiftdetr.models import PostProcess, build_model

if TYPE_CHECKING:
    from swiftdetr.config import ModelConfig


class ModelContext:
    """Lightweight model wrapper returned by ``SwiftDetr.get_model()``.

    Args:
        model: The underlying ``nn.Module`` (SwiftDetr LWDETR instance).
        postprocess: PostProcess for converting raw outputs to bounding boxes.
        device: Device the model lives on.
        resolution: Input resolution (square side length in pixels).
        args: Namespace produced by ``_namespace_from_configs``.
        class_names: Optional list of class name strings from a checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        postprocess: PostProcess,
        device: torch.device,
        resolution: int,
        args: Any,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.postprocess = postprocess
        self.device = device
        self.resolution = resolution
        self.args = args
        self.class_names = class_names
        self.inference_model = None

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Reinitialize the detection head for a new number of classes.

        Args:
            num_classes: New number of output classes (including background).
        """
        self.model.reinitialize_detection_head(num_classes)
        self.args.num_classes = num_classes


def _build_model_context(model_config: "ModelConfig") -> ModelContext:
    """Build a ModelContext from a ModelConfig.

    Replicates the construction logic: builds the nn.Module, optionally loads
    pretrain weights. The model stays on CPU; the caller moves it to the target
    device on first use.

    Args:
        model_config: Architecture configuration.

    Returns:
        ``ModelContext`` with the model on CPU.
    """
    from swiftdetr._namespace import _namespace_from_configs
    from swiftdetr.models.weights import load_pretrain_weights

    dummy_train_config = TrainConfig(dataset_dir=".", output_dir=".")
    args = _namespace_from_configs(model_config, dummy_train_config)
    nn_model = build_model(args)

    class_names: List[str] = []
    if model_config.pretrain_weights is not None:
        class_names = load_pretrain_weights(nn_model, model_config, for_inference=True)
        if hasattr(args, "num_classes") and getattr(args, "num_classes") != model_config.num_classes:
            args.num_classes = model_config.num_classes

    device = torch.device(args.device)
    postprocess = PostProcess(num_select=args.num_select)

    return ModelContext(
        model=nn_model,
        postprocess=postprocess,
        device=device,
        resolution=model_config.resolution,
        args=args,
        class_names=class_names or None,
    )
