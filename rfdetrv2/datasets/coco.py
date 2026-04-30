# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pycocotools.mask as coco_mask
import torch
import torch.utils.data
import torchvision
from PIL import Image

import rfdetrv2.datasets.transforms as T

logger = logging.getLogger(__name__)


def compute_multi_scale_scales(resolution: int, expanded_scales: bool = False, patch_size: int = 16, num_windows: int = 4) -> List[int]:
    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]  # ensure minimum image size
    return proposed_scales


def convert_coco_poly_to_mask(segmentations: List[Any], height: int, width: int) -> torch.Tensor:
    """Convert polygon segmentation to a binary mask tensor of shape [N, H, W].
    Requires pycocotools.
    """
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder: Union[str, Path], ann_file: Union[str, Path], transforms: Optional[Any], include_masks: bool = False) -> None:
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.include_masks = include_masks
        # Raw COCO category_id → contiguous 0..K-1 (SetCriterion / scatter expect this range).
        cat_ids = sorted(int(x) for x in self.coco.getCatIds())
        self._cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}
        self.prepare = ConvertCoco(
            include_masks=include_masks,
            valid_category_ids=set(cat_ids),
        )

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if "labels" in target and target["labels"].numel() > 0:
            labels = target["labels"]
            mapped = torch.tensor(
                [self._cat_id_to_label[int(x.item())] for x in labels],
                dtype=labels.dtype,
                device=labels.device,
            )
            target["labels"] = mapped
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco(object):

    def __init__(
        self,
        include_masks: bool = False,
        valid_category_ids: Optional[set] = None,
    ) -> None:
        self.include_masks = include_masks
        self.valid_category_ids = valid_category_ids

    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        if self.valid_category_ids is not None:
            anno = [
                obj for obj in anno
                if int(obj["category_id"]) in self.valid_category_ids
            ]

        if len(anno) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": image_id,
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "orig_size": torch.as_tensor([int(h), int(w)]),
                "size": torch.as_tensor([int(h), int(w)]),
            }
            if self.include_masks:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8).bool()
            return image, target

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # add segmentation masks if requested, otherwise ensure consistent key when include_masks=True
        if self.include_masks:
            if len(anno) > 0 and 'segmentation' in anno[0]:
                segmentations = [obj.get("segmentation", []) for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                if masks.numel() > 0:
                    target["masks"] = masks[keep]
                else:
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target["masks"] = target["masks"].bool()

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set: str, resolution: int, multi_scale: bool = False, expanded_scales: bool = False, skip_random_resize: bool = False, patch_size: int = 16, num_windows: int = 4) -> T.Compose:

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([resolution], max_size=1333),
            normalize,
        ])
    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_square_div_64(image_set: str, resolution: int, multi_scale: bool = False, expanded_scales: bool = False, skip_random_resize: bool = False, patch_size: int = 16, num_windows: int = 4) -> T.Compose:

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.SquareResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.SquareResize(scales),
                ]),
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])
    if image_set == 'test':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])
    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def _json_has_categories(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("categories"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def resolve_coco_annotation_path_for_categories(root: Union[str, Path]) -> Optional[Path]:
    """Find a COCO JSON that contains ``categories`` (train, val, or any split).

    Used to infer class names and ``num_classes`` without hardcoding MS-COCO labels.
    """
    root = Path(root)
    candidates = [
        root / "train" / "_annotations.coco.json",
        root / "annotations_VisDrone_train.json",
        root / "annotations_VisDrone_val.json",
        root / "annotations" / "instances_train2017.json",
        root / "annotations" / "instances_val2017.json",
        root / "val" / "_annotations.coco.json",
        root / "valid" / "_annotations.coco.json",
        root / "test" / "_annotations.coco.json",
    ]
    for p in candidates:
        if p.is_file() and _json_has_categories(p):
            return p
    ann_dir = root / "annotations"
    if ann_dir.is_dir():
        for p in sorted(ann_dir.glob("*.json")):
            if p.is_file() and _json_has_categories(p):
                return p
    return None


def resolve_coco_train_annotation_path(root: Union[str, Path]) -> Optional[Path]:
    """Return a train-split COCO JSON path if it exists (narrower than categories resolver)."""
    root = Path(root)
    candidates = [
        root / "train" / "_annotations.coco.json",
        root / "annotations_VisDrone_train.json",
        root / "annotations" / "instances_train2017.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return resolve_coco_annotation_path_for_categories(root)


def infer_coco_num_classes_and_names(
    root: Union[str, Path],
) -> Optional[Tuple[List[str], int, List[int]]]:
    """Read categories from the train COCO JSON.

    Returns ``(class_names, num_classes, label_to_cat_id)`` where ``num_classes`` is **K**
    (the number of foreground classes). Training uses contiguous labels ``0..K-1`` via
    :class:`CocoDetection`; ``label_to_cat_id[i]`` is the COCO ``category_id`` for index ``i``.
    """
    ann_path = resolve_coco_annotation_path_for_categories(root)
    if ann_path is None:
        return None
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            anns = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    categories = anns.get("categories") or []
    if not categories:
        return None
    categories = sorted(categories, key=lambda c: int(c["id"]))
    class_names = [str(c.get("name", "")) for c in categories]
    label_to_cat_id = [int(c["id"]) for c in categories if "id" in c]
    if not label_to_cat_id:
        return None
    num_classes = len(categories)
    return class_names, num_classes, label_to_cat_id


def _resolve_build_coco_ann_and_images(root: Path, split: str) -> Tuple[Path, Path]:
    """Pick (image_folder, annotation_json) for train / val / test layouts (COCO, VisDrone, Roboflow, etc.)."""
    ann_candidates = {
        "train": [
            root / "train" / "_annotations.coco.json",
            root / "annotations_VisDrone_train.json",
            root / "annotations" / "instances_train2017.json",
            root / "annotations" / "train.json",
        ],
        "val": [
            root / "val" / "_annotations.coco.json",
            root / "valid" / "_annotations.coco.json",
            root / "annotations_VisDrone_val.json",
            root / "annotations" / "instances_val2017.json",
            root / "annotations" / "val.json",
        ],
        "test": [
            root / "test" / "_annotations.coco.json",
            root / "val" / "_annotations.coco.json",
            root / "valid" / "_annotations.coco.json",
            root / "annotations_VisDrone_val.json",
            root / "annotations" / "instances_val2017.json",
            root / "annotations" / "val.json",
        ],
    }
    img_candidates = {
        "train": [
            root / "train",
            root / "train2017",
            root / "images" / "train",
            root / "images",
        ],
        "val": [
            root / "val",
            root / "val2017",
            root / "images" / "val",
            root / "valid",
            root / "images",
        ],
        "test": [
            root / "test",
            root / "val2017",
            root / "images" / "test",
            root / "valid",
            root / "images",
        ],
    }

    ann_file: Optional[Path] = None
    for p in ann_candidates.get(split, []):
        if p.is_file() and _json_has_categories(p):
            ann_file = p
            break
    if ann_file is None:
        ann_dir = root / "annotations"
        if ann_dir.is_dir():
            for p in sorted(ann_dir.glob("*.json")):
                if p.is_file() and _json_has_categories(p):
                    name_l = p.name.lower()
                    if split == "train" and ("train" in name_l or "instances_train" in name_l):
                        ann_file = p
                        break
                    if split in ("val", "test") and ("val" in name_l or "instances_val" in name_l):
                        ann_file = p
                        break
            if ann_file is None:
                for p in sorted(ann_dir.glob("*.json")):
                    if p.is_file() and _json_has_categories(p):
                        ann_file = p
                        break

    if ann_file is None:
        raise FileNotFoundError(
            f"Cannot find a COCO annotation JSON with categories for split={split!r} under {root}. "
            f"Tried: {[str(p) for p in ann_candidates.get(split, [])]}"
        )

    img_folder: Optional[Path] = None
    for p in img_candidates.get(split, []):
        if p.is_dir():
            img_folder = p
            break
    if img_folder is None:
        img_folder = root

    return img_folder, ann_file


def build_coco(image_set: str, args: Any, resolution: int) -> CocoDetection:
    """Build a COCO-format dataset; annotation and image paths are auto-detected."""
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    split = image_set.split("_")[0]
    img_folder, ann_file = _resolve_build_coco_ann_and_images(root, split)
    logger.info("build_coco split=%s images=%s ann=%s", split, img_folder, ann_file)

    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", False)
    skip_random_resize = not getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", 16)
    num_windows = getattr(args, "num_windows", 4)

    transform_fn = (
        make_coco_transforms_square_div_64 if square_resize_div_64 else make_coco_transforms
    )
    transforms = transform_fn(
        image_set,
        resolution,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        skip_random_resize=skip_random_resize,
        patch_size=patch_size,
        num_windows=num_windows,
    )
    return CocoDetection(
        img_folder,
        ann_file,
        transforms=transforms,
        include_masks=include_masks,
    )

def build_roboflow_from_coco(image_set: str, args: Any, resolution: int) -> CocoDetection:
    """Build a Roboflow COCO-format dataset.

    This uses Roboflow's standard directory structure
    (train/valid/test folders with _annotations.coco.json).
    """
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    PATHS = {
        "train": (root / "images", root / "train" / "_annotations.coco.json"),
        "val": (root /  "images", root / "valid" / "_annotations.coco.json"),
        "test": (root / "images", root / "valid" / "_annotations.coco.json"),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", False)
    do_random_resize_via_padding = getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", 16)
    num_windows = getattr(args, "num_windows", 4)

    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            resolution,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            skip_random_resize=not do_random_resize_via_padding,
            patch_size=patch_size,
            num_windows=num_windows
        ), include_masks=include_masks)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            resolution,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            skip_random_resize=not do_random_resize_via_padding,
            patch_size=patch_size,
            num_windows=num_windows
        ), include_masks=include_masks)
    return dataset
