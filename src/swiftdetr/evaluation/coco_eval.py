# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""COCO evaluator for ONNX/TRT export benchmarking.

Provides :class:`CocoEvaluator` used by :mod:`swiftdetr.export.benchmark` to
compute mAP during ONNX and TensorRT inference benchmarks.

Mostly copy-paste from
https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
"""

import contextlib
import copy
import os
from typing import Any, Optional

import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from swiftdetr.evaluation.f1_sweep import sweep_confidence_thresholds
from swiftdetr.util.distributed import all_gather
from swiftdetr.util.logger import get_logger

logger = get_logger()


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [x1, y1, x2, y2] to [x1, y1, w, h]."""
    boxes = boxes.copy()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    return boxes


class CocoEvaluator:
    """COCO evaluator that works in distributed mode."""

    def __init__(self, coco_gt: COCO, iou_types: list[str], max_dets: int = 100) -> None:
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.max_dets = max_dets
        # label2cat maps contiguous model label indices back to original COCO category_ids.
        # Set by CocoDetection when cat2label remapping is active; None otherwise.
        self.label2cat: dict[int, int] | None = getattr(coco_gt, "label2cat", None)

        self.iou_types = iou_types
        self.coco_eval: dict[str, COCOeval] = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].params.maxDets = [1, 10, max_dets]

        self.img_ids: list[int] = []
        self.eval_imgs: dict[str, list[Any]] = {k: [] for k in iou_types}
        self.cat_ids = set(coco_gt.cats.keys())
        self._prefer_raw_category_ids = False

    def _resolve_category_id(self, label: int, use_raw_category_ids: bool) -> int | None:
        """Resolve a predicted label to a COCO category_id."""
        if use_raw_category_ids:
            return label if label in self.cat_ids else None
        if self.label2cat is not None and label in self.label2cat:
            return self.label2cat[label]
        if label in self.cat_ids:
            return label
        return None

    def _should_use_raw_category_ids(self, labels: list[int]) -> bool:
        """Detect whether model predictions are already raw COCO category IDs."""
        if self.label2cat is None:
            return True
        if self._prefer_raw_category_ids:
            return True
        uses_raw_ids = list(self.label2cat.keys()) == list(self.label2cat.values())
        if uses_raw_ids:
            self._prefer_raw_category_ids = True
            return True
        return False

    def update(self, predictions: dict[int, Any]) -> None:
        """Accumulate per-image predictions."""
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self, distribute: bool = True) -> None:
        """Merge eval results across distributed processes.

        Args:
            distribute: If True, use all_gather to collect results from all
                DDP ranks (standard distributed eval). If False, merge only
                local data — use this when only one rank runs evaluation
                (e.g. EMA eval on rank 0 only).
        """
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            if distribute:
                create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])
            else:
                _local_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self) -> None:
        """Accumulate per-image evaluation results into mean metrics."""
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self) -> None:
        """Print and log COCO summary statistics."""
        for iou_type, coco_eval in self.coco_eval.items():
            logger.info("IoU metric: {}".format(iou_type))
            patched_pycocotools_summarize(coco_eval)

    def prepare(self, predictions: dict[int, Any], iou_type: str) -> list[dict[str, Any]]:
        """Convert predictions to COCO format for the given iou_type."""
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions: dict[int, Any]) -> list[dict[str, Any]]:
        """Format bounding-box predictions as COCO result dicts."""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = _xyxy_to_xywh(boxes.cpu().numpy()).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            use_raw_category_ids = self._should_use_raw_category_ids(labels)
            for k, box in enumerate(boxes):
                category_id = self._resolve_category_id(labels[k], use_raw_category_ids)
                if category_id is None:
                    continue
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": category_id,
                        "bbox": box,
                        "score": scores[k],
                    }
                )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions: dict[int, Any]) -> list[dict[str, Any]]:
        """Format segmentation mask predictions as COCO result dicts."""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            use_raw_category_ids = self._should_use_raw_category_ids(labels)

            rles = [
                mask_util.encode(np.array(mask.cpu()[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            for k, rle in enumerate(rles):
                category_id = self._resolve_category_id(labels[k], use_raw_category_ids)
                if category_id is None:
                    continue
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": category_id,
                        "segmentation": rle,
                        "score": scores[k],
                    }
                )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions: dict[int, Any]) -> list[dict[str, Any]]:
        """Format keypoint predictions as COCO result dicts."""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = _xyxy_to_xywh(boxes.cpu().numpy()).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()
            use_raw_category_ids = self._should_use_raw_category_ids(labels)
            for k, keypoint in enumerate(keypoints):
                category_id = self._resolve_category_id(labels[k], use_raw_category_ids)
                if category_id is None:
                    continue
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": category_id,
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                )
        return coco_results


def merge(img_ids: list[int], eval_imgs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Merge distributed per-image evaluation results."""
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids: list[int] = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids_arr = np.array(merged_img_ids)
    merged_eval_imgs_arr = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids_arr, idx = np.unique(merged_img_ids_arr, return_index=True)
    merged_eval_imgs_arr = merged_eval_imgs_arr[..., idx]

    return merged_img_ids_arr, merged_eval_imgs_arr


def create_common_coco_eval(coco_eval: COCOeval, img_ids: list[int], eval_imgs: Any) -> None:
    """Populate a COCOeval object with merged distributed results."""
    img_ids_arr, eval_imgs = merge(img_ids, eval_imgs)
    img_ids_list = list(img_ids_arr)
    eval_imgs_list = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs_list
    coco_eval.params.imgIds = img_ids_list
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################
def evaluate(self: COCOeval) -> tuple[list[int], np.ndarray]:
    """Run per-image evaluation and store results in self.evalImgs."""
    p = self.params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        logger.warning("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    category_ids = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        compute_iou = self.computeIoU
    elif p.iouType == "keypoints":
        compute_iou = self.computeOks
    self.ious = {(imgId, catId): compute_iou(imgId, catId) for imgId in p.imgIds for catId in category_ids}

    evaluate_image = self.evaluateImg
    max_det = p.maxDets[-1]
    eval_images = [
        evaluate_image(imgId, catId, areaRng, max_det)
        for catId in category_ids
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    eval_images = np.asarray(eval_images).reshape(len(category_ids), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    return p.imgIds, eval_images


#################################################################
# From pycocotools, patched first _summarize() call to use
# maxDets[-1] instead of hardcoded 100.
#################################################################
def patched_pycocotools_summarize(self: COCOeval) -> None:
    """Compute and display summary metrics for evaluation results."""

    def _summarize(ap: int = 1, iou_thr: float | None = None, area_rng: str = "all", max_dets: int = 100) -> float:
        p = self.params
        log_template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
        title_str = "Average Precision" if ap == 1 else "Average Recall"
        type_str = "(AP)" if ap == 1 else "(AR)"
        iou_str = (
            "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) if iou_thr is None else "{:0.2f}".format(iou_thr)
        )

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_rng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_dets]
        if ap == 1:
            s = self.eval["precision"]
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        mean_s = -1 if len(s[s > -1]) == 0 else float(np.mean(s[s > -1]))
        logger.info(log_template.format(title_str, type_str, iou_str, area_rng, max_dets, mean_s))
        return mean_s

    def _summarizeDets() -> np.ndarray:  # noqa: N802
        stats = np.zeros((12,))
        stats[0] = _summarize(1, max_dets=self.params.maxDets[2])
        stats[1] = _summarize(1, iou_thr=0.5, max_dets=self.params.maxDets[2])
        stats[2] = _summarize(1, iou_thr=0.75, max_dets=self.params.maxDets[2])
        stats[3] = _summarize(1, area_rng="small", max_dets=self.params.maxDets[2])
        stats[4] = _summarize(1, area_rng="medium", max_dets=self.params.maxDets[2])
        stats[5] = _summarize(1, area_rng="large", max_dets=self.params.maxDets[2])
        stats[6] = _summarize(0, max_dets=self.params.maxDets[0])
        stats[7] = _summarize(0, max_dets=self.params.maxDets[1])
        stats[8] = _summarize(0, max_dets=self.params.maxDets[2])
        stats[9] = _summarize(0, area_rng="small", max_dets=self.params.maxDets[2])
        stats[10] = _summarize(0, area_rng="medium", max_dets=self.params.maxDets[2])
        stats[11] = _summarize(0, area_rng="large", max_dets=self.params.maxDets[2])
        return stats

    def _summarizeKps() -> np.ndarray:  # noqa: N802
        stats = np.zeros((10,))
        stats[0] = _summarize(1, max_dets=20)
        stats[1] = _summarize(1, max_dets=20, iou_thr=0.5)
        stats[2] = _summarize(1, max_dets=20, iou_thr=0.75)
        stats[3] = _summarize(1, max_dets=20, area_rng="medium")
        stats[4] = _summarize(1, max_dets=20, area_rng="large")
        stats[5] = _summarize(0, max_dets=20)
        stats[6] = _summarize(0, max_dets=20, iou_thr=0.5)
        stats[7] = _summarize(0, max_dets=20, iou_thr=0.75)
        stats[8] = _summarize(0, max_dets=20, area_rng="medium")
        stats[9] = _summarize(0, max_dets=20, area_rng="large")
        return stats

    if not self.eval:
        raise Exception("Please run accumulate() first")
    iou_type = self.params.iouType
    if iou_type == "segm" or iou_type == "bbox":
        summarize = _summarizeDets
    elif iou_type == "keypoints":
        summarize = _summarizeKps
    self.stats = summarize()


def _local_coco_eval(coco_eval: COCOeval, img_ids: list[int], eval_imgs: Any) -> None:
    """Populate a COCOeval object from local (single-rank) data without all_gather."""
    eval_imgs_flat = list(eval_imgs.flatten())
    img_ids_sorted = list(np.unique(np.array(img_ids)))
    coco_eval.evalImgs = eval_imgs_flat
    coco_eval.params.imgIds = img_ids_sorted
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def coco_extended_metrics(coco_eval: COCOeval) -> dict[str, Any]:
    """Compute per-class AP table and F1/precision/recall from a finished COCOeval.

    Must be called after ``coco_eval.accumulate()``.

    Returns a dict with keys:
        - ``"mAP_50_95"``, ``"mAP_50"``, ``"mAP_75"``, ``"mAR"``
        - ``"F1"``, ``"precision"``, ``"recall"``
        - ``"AP/{class_name}"`` per class
        - ``"ap_per_class_markdown"`` — markdown table string
    """
    iou50_idx = int(np.argwhere(np.isclose(coco_eval.params.iouThrs, 0.50)).item())
    cat_ids = coco_eval.params.catIds
    area_idx = 0
    maxdet_idx = 2

    # Unflatten evalImgs into nested dict: cat_id → area_rng → img_id → e
    evalImgs_unflat: dict = {}
    for e in coco_eval.evalImgs:
        if e is None:
            continue
        cid = e["category_id"]
        arng = tuple(e["aRng"])
        iid = e["image_id"]
        evalImgs_unflat.setdefault(cid, {}).setdefault(arng, {})[iid] = e

    area_rng_all = tuple(coco_eval.params.areaRng[area_idx])

    per_class_data = []
    for cid in cat_ids:
        dt_scores: list = []
        dt_matches: list = []
        dt_ignore: list = []
        total_gt = 0
        for img_id in coco_eval.params.imgIds:
            e = evalImgs_unflat.get(cid, {}).get(area_rng_all, {}).get(img_id)
            if e is None:
                continue
            gt_ignore = e["gtIgnore"]
            total_gt += sum(1 for ig in gt_ignore if not ig)
            for d in range(len(e["dtIds"])):
                dt_scores.append(e["dtScores"][d])
                dt_matches.append(e["dtMatches"][iou50_idx, d])
                dt_ignore.append(e["dtIgnore"][iou50_idx, d])
        per_class_data.append({
            "scores": np.array(dt_scores, dtype=np.float32),
            "matches": np.array(dt_matches),
            "ignore": np.array(dt_ignore, dtype=bool),
            "total_gt": total_gt,
        })

    classes_with_gt = [k for k in range(len(cat_ids)) if per_class_data[k]["total_gt"] > 0]
    conf_thresholds = np.linspace(0.0, 1.0, 101)
    sweep_results = sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt)
    best = max(sweep_results, key=lambda x: x["macro_f1"])

    stats = coco_eval.stats
    mAP_50_95 = float(stats[0])
    mAP_50    = float(stats[1])
    mAP_75    = float(stats[2])
    mAR       = float(stats[8])

    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}

    out: dict[str, Any] = {
        "mAP_50_95": mAP_50_95,
        "mAP_50":    mAP_50,
        "mAP_75":    mAP_75,
        "mAR":       mAR,
        "F1":        float(best["macro_f1"]),
        "precision": float(best["macro_precision"]),
        "recall":    float(best["macro_recall"]),
    }

    # Per-class AP (50:95 and 50)
    per_class_rows = []
    for k, cid in enumerate(cat_ids):
        p_slice = coco_eval.eval["precision"][:, :, k, area_idx, maxdet_idx]
        p_masked = np.where(p_slice > -1, p_slice, np.nan)
        ap_50_95 = float(np.nanmean(np.nanmean(p_masked, axis=1)))
        ap_50 = float(np.nanmean(p_masked[iou50_idx]))
        if np.isnan(ap_50_95) or np.isnan(ap_50):
            continue
        name = cat_id_to_name.get(int(cid), str(cid))
        out[f"AP/{name}"] = ap_50_95
        per_class_rows.append({"class": name, "ap50_95": ap_50_95, "ap50": ap_50})

    per_class_rows.sort(key=lambda r: r["ap50_95"], reverse=True)
    lines = ["| rank | class | ap50_95 | ap50 |", "| --- | --- | ---: | ---: |"]
    for rank, row in enumerate(per_class_rows, 1):
        lines.append(f"| {rank} | {row['class']} | {row['ap50_95']:.4f} | {row['ap50']:.4f} |")
    lines.append(f"| all | all | {mAP_50_95:.4f} | {mAP_50:.4f} |")
    out["ap_per_class_markdown"] = "\n".join(lines)

    return out
