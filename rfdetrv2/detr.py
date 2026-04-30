# ------------------------------------------------------------------------
# SwiftDetr high-level API
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import glob
import json
import os
from collections import defaultdict
from copy import deepcopy
from logging import getLogger
from typing import List, Optional, Union

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image

try:
    torch.set_float32_matmul_precision('high')
except:
    pass

from rfdetrv2.config import (
    ModelConfig,
    SwiftDetrBaseConfig,
    SwiftDetrSmallConfig,
    SwiftDetrTinyConfig,
    TrainConfig,
)
from rfdetrv2.datasets.coco import infer_coco_num_classes_and_names
from rfdetrv2.main import Model
from rfdetrv2.util.coco_classes import COCO_CLASSES
from rfdetrv2.util.metrics import MetricsPlotSink, MetricsTensorBoardSink, MetricsWandBSink

logger = getLogger(__name__)
class SwiftDetr:
    """
    Base SwiftDetr API: training, inference, optimization, and deployment helpers.
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = None

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        # self.maybe_download_pretrain_weights()
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)

        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None

    def maybe_download_pretrain_weights(self):
        """
        Download pre-trained weights if they are not already downloaded.
        """
        # download_pretrain_weights(self.model_config.pretrain_weights)
        return 

    def get_model_config(self, **kwargs):
        """
        Retrieve the configuration parameters used by the model.
        """
        return ModelConfig(**kwargs)

    def train(self, **kwargs):
        """
        Train a SwiftDetr model.
        """
        config = self.get_train_config(**kwargs)
        self.train_from_config(config, **kwargs)

    def optimize_for_inference(self, compile=True, batch_size=1, dtype=torch.float32):
        self.remove_optimized_model()

        self.model.inference_model = deepcopy(self.model.model)
        self.model.inference_model.eval()
        self.model.inference_model.export()

        self._optimized_resolution = self.model.resolution
        self._is_optimized_for_inference = True

        self.model.inference_model = self.model.inference_model.to(dtype=dtype)
        self._optimized_dtype = dtype

        if compile:
            self.model.inference_model = torch.jit.trace(
                self.model.inference_model,
                torch.randn(
                    batch_size, 3, self.model.resolution, self.model.resolution,
                    device=self.model.device,
                    dtype=dtype
                )
            )
            self._optimized_has_been_compiled = True
            self._optimized_batch_size = batch_size

    def remove_optimized_model(self):
        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_half = False

    def export(self, **kwargs):
        """
        Export your model to an ONNX file.

        See [the ONNX export documentation](https://rfdetr.roboflow.com/learn/export/) for more information.
        """
        self.model.export(**kwargs)

    @staticmethod
    def _load_classes(dataset_dir) -> List[str]:
        """Load class names from a COCO or YOLO dataset directory."""
        coco_path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
        if os.path.exists(coco_path):
            with open(coco_path, "r") as f:
                anns = json.load(f)
            class_names = [c["name"] for c in anns["categories"] if c["supercategory"] != "none"]
            return class_names

        # list all YAML files in the folder
        yaml_paths = glob.glob(os.path.join(dataset_dir, "*.yaml")) + glob.glob(os.path.join(dataset_dir, "*.yml"))
        # any YAML file starting with data e.g. data.yaml, dataset.yaml
        yaml_data_files = [yp for yp in yaml_paths if os.path.basename(yp).startswith("data")]
        if len(yaml_data_files) == 1:
            yaml_path = yaml_data_files[0]
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            if "names" in data:
                if isinstance(data["names"], dict):
                    return [data["names"][i] for i in sorted(data["names"].keys())]
                return data["names"]
            else:
                raise ValueError(f"Found {yaml_path} but it does not contain 'names' field.")
        elif len(yaml_data_files) > 1:
            raise ValueError(f"Found multiple YAML files starting with 'data' in {dataset_dir}: {yaml_data_files}. "
                             "Please rename one of them to avoid conflicts.")

        raise FileNotFoundError(
            f"Could not find class names in {dataset_dir}. "
            "Checked for COCO (train/_annotations.coco.json) and YOLO (data.yaml, data.yml) styles."
        )

    def train_from_config(self, config: TrainConfig, **kwargs):
        # ``coco_path`` is required by ``build_coco`` but not always passed (TrainConfig may omit it).
        kwargs = dict(kwargs)
        if kwargs.get("coco_path") is None:
            kwargs["coco_path"] = getattr(config, "coco_path", None) or config.dataset_dir

        label_to_cat_id: Optional[List[int]] = None
        if config.dataset_file == "roboflow":
            coco_ann = os.path.join(config.dataset_dir, "train", "_annotations.coco.json")
            if os.path.exists(coco_ann):
                with open(coco_ann, "r") as f:
                    anns = json.load(f)
                cats = sorted(anns.get("categories", []), key=lambda c: int(c["id"]))
                cats = [c for c in cats if c.get("supercategory") != "none"]
                class_names = [c["name"] for c in cats]
                label_to_cat_id = [int(c["id"]) for c in cats]
                num_classes = len(class_names)
            else:
                class_names = self._load_classes(config.dataset_dir)
                num_classes = len(class_names)
            self.model.class_names = class_names
        elif config.dataset_file == "coco":
            # K foreground classes; CocoDetection remaps raw category_id → 0..K-1.
            root = getattr(config, "coco_path", None) or config.dataset_dir
            inferred = infer_coco_num_classes_and_names(root)
            if inferred is not None:
                class_names, num_classes, label_to_cat_id = inferred
                self.model.class_names = class_names
                logger.info(
                    "dataset_file=coco: num_classes=%d (K foreground) from annotations under %s",
                    num_classes,
                    root,
                )
            else:
                raise RuntimeError(
                    f"Could not read class names from any COCO JSON under {root}. "
                    "Expected a file with a \"categories\" list, e.g. "
                    "train/_annotations.coco.json, annotations_VisDrone_train.json, "
                    "annotations/instances_train2017.json, or val/_annotations.coco.json."
                )
        else:
            raise ValueError(f"Invalid dataset file: {config.dataset_file}")

        # ``build_model`` / ``LWDETR`` use ``args.num_classes + 1`` logits; head resize must match.
        if self.model_config.num_classes != num_classes:
            self.model.reinitialize_detection_head(num_classes + 1)

        train_config = config.dict()
        model_config = self.model_config.dict()
        model_config.pop("num_classes")
        if "class_names" in model_config:
            model_config.pop("class_names")

        if "class_names" in train_config and train_config["class_names"] is None:
            train_config["class_names"] = class_names

        for k, v in train_config.items():
            if k in model_config:
                model_config.pop(k)
            if k in kwargs:
                kwargs.pop(k)

        all_kwargs = {**model_config, **train_config, **kwargs, "num_classes": num_classes}
        if label_to_cat_id is not None:
            all_kwargs["label_to_cat_id"] = label_to_cat_id
        # ``train_config`` may contain ``coco_path=None``; that would win over kwargs after the pop loop.
        if all_kwargs.get("coco_path") is None:
            all_kwargs["coco_path"] = getattr(config, "coco_path", None) or config.dataset_dir

        metrics_plot_sink = MetricsPlotSink(output_dir=config.output_dir)
        self.callbacks["on_fit_epoch_end"].append(metrics_plot_sink.update)
        self.callbacks["on_train_end"].append(metrics_plot_sink.save)

        if config.tensorboard:
            metrics_tensor_board_sink = MetricsTensorBoardSink(output_dir=config.output_dir)
            self.callbacks["on_fit_epoch_end"].append(metrics_tensor_board_sink.update)
            self.callbacks["on_train_end"].append(metrics_tensor_board_sink.close)

        if config.wandb:
            metrics_wandb_sink = MetricsWandBSink(
                output_dir=config.output_dir,
                project=config.project,
                run=config.run,
                config=config.model_dump()
            )
            self.callbacks["on_fit_epoch_end"].append(metrics_wandb_sink.update)
            self.callbacks["on_train_end"].append(metrics_wandb_sink.close)

        if config.early_stopping:
            from rfdetr.util.early_stopping import EarlyStoppingCallback
            early_stopping_callback = EarlyStoppingCallback(
                model=self.model,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                use_ema=config.early_stopping_use_ema,
                segmentation_head=config.segmentation_head
            )
            self.callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)

        self.model.train(
            **all_kwargs,
            callbacks=self.callbacks,
        )

    def get_train_config(self, **kwargs):
        """
        Retrieve the configuration parameters that will be used for training.
        """
        return TrainConfig(**kwargs)

    def get_model(self, config: ModelConfig):
        """Build the detection Model from configuration."""
        return Model(**config.dict())

    # Get class_names from the model
    @property
    def class_names(self):
        """
        Class names for inference / visualization.

        After training, names come from the dataset (``Model.class_names`` list and optional
        ``label_to_cat_id`` on ``Model.args``). Keys are COCO ``category_id`` when
        ``label_to_cat_id`` is set; otherwise contiguous ``1..K``.
        """
        if hasattr(self.model, "class_names") and self.model.class_names:
            raw = self.model.class_names
            if isinstance(raw, dict):
                return raw
            names = list(raw)
            args = getattr(self.model, "args", None)
            lid = getattr(args, "label_to_cat_id", None) if args is not None else None
            if lid is not None and len(lid) == len(names):
                return {int(cid): str(n) for cid, n in zip(lid, names)}
            return {i + 1: str(n) for i, n in enumerate(names)}

        return COCO_CLASSES

    def predict(
        self,
        images: Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
        threshold: float = 0.5,
        **kwargs,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        """Performs object detection on the input images and returns bounding box
        predictions.

        This method accepts a single image or a list of images in various formats
        (file path, PIL Image, NumPy array, or torch.Tensor). The images should be in
        RGB channel order. If a torch.Tensor is provided, it must already be normalized
        to values in the [0, 1] range and have the shape (C, H, W).

        Args:
            images (Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]]):
                A single image or a list of images to process. Images can be provided
                as file paths, PIL Images, NumPy arrays, or torch.Tensors.
            threshold (float, optional):
                The minimum confidence score needed to consider a detected bounding box valid.
            **kwargs:
                Additional keyword arguments.

        Returns:
            Union[sv.Detections, List[sv.Detections]]: A single or multiple Detections
                objects, each containing bounding box coordinates, confidence scores,
                and class IDs.
        """
        if not self._is_optimized_for_inference and not self._has_warned_about_not_being_optimized_for_inference:
            logger.warning(
                "Model is not optimized for inference. "
                "Latency may be higher than expected. "
                "You can optimize the model for inference by calling model.optimize_for_inference()."
            )
            self._has_warned_about_not_being_optimized_for_inference = True

            self.model.model.eval()

        if not isinstance(images, list):
            images = [images]

        orig_sizes = []
        processed_images = []

        for img in images:

            if isinstance(img, str):
                img = Image.open(img).convert("RGB")

            if not isinstance(img, torch.Tensor):
                img = F.to_tensor(img)

            if (img > 1).any():
                raise ValueError(
                    "Image has pixel values above 1. Please ensure the image is "
                    "normalized (scaled to [0, 1])."
                )
            if img.shape[0] != 3:
                raise ValueError(
                    f"Invalid image shape. Expected 3 channels (RGB), but got "
                    f"{img.shape[0]} channels."
                )
            img_tensor = img

            h, w = img_tensor.shape[1:]
            orig_sizes.append((h, w))

            img_tensor = img_tensor.to(self.model.device)
            img_tensor = F.normalize(img_tensor, self.means, self.stds)
            img_tensor = F.resize(img_tensor, (self.model.resolution, self.model.resolution))

            processed_images.append(img_tensor)

        batch_tensor = torch.stack(processed_images)

        if self._is_optimized_for_inference:
            if self._optimized_resolution != batch_tensor.shape[2]:
                # this could happen if someone manually changes self.model.resolution after optimizing the model
                raise ValueError(f"Resolution mismatch. "
                                 f"Model was optimized for resolution {self._optimized_resolution}, "
                                 f"but got {batch_tensor.shape[2]}. "
                                 "You can explicitly remove the optimized model by calling model.remove_optimized_model().")
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != batch_tensor.shape[0]:
                    raise ValueError(f"Batch size mismatch. "
                                     f"Optimized model was compiled for batch size {self._optimized_batch_size}, "
                                     f"but got {batch_tensor.shape[0]}. "
                                     "You can explicitly remove the optimized model by calling model.remove_optimized_model(). "
                                     "Alternatively, you can recompile the optimized model for a different batch size "
                                     "by calling model.optimize_for_inference(batch_size=<new_batch_size>).")

        with torch.no_grad():
            if self._is_optimized_for_inference:
                predictions = self.model.inference_model(batch_tensor.to(dtype=self._optimized_dtype))
            else:
                predictions = self.model.model(batch_tensor)
            if isinstance(predictions, tuple):
                return_predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0],
                }
                if len(predictions) == 3:
                    return_predictions["pred_masks"] = predictions[2]
                predictions = return_predictions
            target_sizes = torch.tensor(orig_sizes, device=self.model.device)
            results = self.model.postprocess(predictions, target_sizes=target_sizes)

        detections_list = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            if "masks" in result:
                masks = result["masks"]
                masks = masks[keep]

                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                    mask=masks.squeeze(1).cpu().numpy(),
                )
            else:
                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                )

            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]

    def deploy_to_roboflow(self, workspace: str, project_id: str, version: str, api_key: str = None, size: str = None):
        """
        Deploy the trained RF-DETR model to Roboflow.

        Deploying with Roboflow will create a Serverless API to which you can make requests.

        You can also download weights into a Roboflow Inference deployment for use in Roboflow Workflows and on-device deployment.

        Args:
            workspace (str): The name of the Roboflow workspace to deploy to.
            project_ids (List[str]): A list of project IDs to which the model will be deployed
            api_key (str, optional): Your Roboflow API key. If not provided,
                it will be read from the environment variable `ROBOFLOW_API_KEY`.
            size (str, optional): The size of the model to deploy. If not provided,
                it will default to the size of the model being trained (e.g., "swiftdetr-base", etc.).
            model_name (str, optional): The name you want to give the uploaded model.
            If not provided, it will default to "<size>-uploaded".
        Raises:
            ValueError: If the `api_key` is not provided and not found in the environment
                variable `ROBOFLOW_API_KEY`, or if the `size` is not set for custom architectures.
        """
        import shutil

        from roboflow import Roboflow
        if api_key is None:
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if api_key is None:
                raise ValueError("Set api_key=<KEY> in deploy_to_roboflow or export ROBOFLOW_API_KEY=<KEY>")


        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace)

        if self.size is None and size is None:
            raise ValueError("Must set size for custom architectures")

        size = self.size or size
        tmp_out_dir = ".roboflow_temp_upload"
        os.makedirs(tmp_out_dir, exist_ok=True)
        outpath = os.path.join(tmp_out_dir, "weights.pt")
        torch.save(
            {
                "model": self.model.model.state_dict(),
                "args": self.model.args
            }, outpath
        )
        project = workspace.project(project_id)
        version = project.version(version)
        version.deploy(
            model_type=size,
            model_path=tmp_out_dir,
            filename="weights.pt"
        )
        shutil.rmtree(tmp_out_dir)



class SwiftDetrBase(SwiftDetr):
    """SwiftDetr-Base: SwiftNet-Base encoder at 640×640."""

    size = "swiftdetr-base"

    def get_model_config(self, **kwargs):
        return SwiftDetrBaseConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class SwiftDetrTiny(SwiftDetr):
    """SwiftDetr-Tiny: SwiftNet-Tiny encoder."""

    size = "swiftdetr-tiny"

    def get_model_config(self, **kwargs):
        return SwiftDetrTinyConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class SwiftDetrSmall(SwiftDetr):
    """SwiftDetr-Small: SwiftNet-Small encoder."""

    size = "swiftdetr-small"

    def get_model_config(self, **kwargs):
        return SwiftDetrSmallConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


# --- Deprecated RF-DETR names (same classes) ---
RFDETRV2 = SwiftDetr
RFDETRBase = SwiftDetrBase
RFDETRNano = SwiftDetrTiny
RFDETRSmall = SwiftDetrSmall
RFDETRLarge = SwiftDetrBase
