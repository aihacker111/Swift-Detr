# # ------------------------------------------------------------------------
# # RF-DETR
# # Copyright (c) 2025 Roboflow. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------
# # Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# # Copyright (c) 2024 Baidu. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Conditional DETR
# # Copyright (c) 2021 Microsoft. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------
# # Copied from DETR (https://github.com/facebookresearch/detr)
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# # ------------------------------------------------------------------------

# """
# Train and eval functions used in main.py
# """
# import math
# import random
# from typing import Iterable

# import torch
# import torch.nn.functional as F

# import rfdetrv2.util.misc as utils
# from rfdetrv2.datasets.coco import compute_multi_scale_scales
# from rfdetrv2.datasets.coco_eval import CocoEvaluator

# try:
#     from torch.amp import GradScaler, autocast
#     DEPRECATED_AMP = False
# except ImportError:
#     from torch.cuda.amp import GradScaler, autocast
#     DEPRECATED_AMP = True
# from typing import Callable, DefaultDict, List

# import numpy as np

# from rfdetrv2.util.misc import NestedTensor


# def get_autocast_args(args):
#     if DEPRECATED_AMP:
#         return {'enabled': args.amp, 'dtype': torch.bfloat16}
#     else:
#         return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}


# def train_one_epoch(
#     model: torch.nn.Module,
#     criterion: torch.nn.Module,
#     lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
#     data_loader: Iterable,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     epoch: int,
#     batch_size: int,
#     max_norm: float = 0,
#     ema_m: torch.nn.Module = None,
#     schedules: dict = {},
#     num_training_steps_per_epoch=None,
#     vit_encoder_num_layers=None,
#     args=None,
#     callbacks: DefaultDict[str, List[Callable]] = None,
#     postprocess=None,
# ):
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     metric_logger.add_meter(
#         "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
#     )
#     header = "Epoch: [{}]".format(epoch)
#     print_freq = 10
#     start_steps = epoch * num_training_steps_per_epoch

#     print("Grad accum steps: ", args.grad_accum_steps)
#     print("Total batch size: ", batch_size * utils.get_world_size())

#     # Add gradient scaler for AMP
#     if DEPRECATED_AMP:
#         scaler = GradScaler(enabled=args.amp)
#     else:
#         scaler = GradScaler('cuda', enabled=args.amp)

#     optimizer.zero_grad()
#     assert batch_size % args.grad_accum_steps == 0
#     sub_batch_size = batch_size // args.grad_accum_steps
#     print("LENGTH OF DATA LOADER:", len(data_loader))
#     for data_iter_step, (samples, targets) in enumerate(
#         metric_logger.log_every(data_loader, print_freq, header)
#     ):
#         it = start_steps + data_iter_step
#         callback_dict = {
#             "step": it,
#             "model": model,
#             "epoch": epoch,
#         }
#         for callback in callbacks["on_train_batch_start"]:
#             callback(callback_dict)
#         if "dp" in schedules:
#             if args.distributed:
#                 model.module.update_drop_path(
#                     schedules["dp"][it], vit_encoder_num_layers
#                 )
#             else:
#                 model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
#         if "do" in schedules:
#             if args.distributed:
#                 model.module.update_dropout(schedules["do"][it])
#             else:
#                 model.update_dropout(schedules["do"][it])

#         if args.multi_scale and not args.do_random_resize_via_padding:
#             scales = compute_multi_scale_scales(args.resolution, args.expanded_scales, args.patch_size, args.num_windows)
#             random.seed(it)
#             scale = random.choice(scales)
#             with torch.no_grad():
#                 samples.tensors = F.interpolate(samples.tensors, size=scale, mode='bilinear', align_corners=False)
#                 samples.mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode='nearest').squeeze(1).bool()

#         rl_reward = None
#         for i in range(args.grad_accum_steps):
#             start_idx = i * sub_batch_size
#             final_idx = start_idx + sub_batch_size
#             new_samples_tensors = samples.tensors[start_idx:final_idx]
#             new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
#             new_samples = new_samples.to(device)
#             new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

#             with autocast(**get_autocast_args(args)):
#                 outputs = model(new_samples, new_targets)
#                 loss_dict = criterion(outputs, new_targets)
#                 weight_dict = criterion.weight_dict
#                 losses = sum(
#                     (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
#                     for k in loss_dict.keys()
#                     if k in weight_dict
#                 )
#                 # RL: weight loss by detection reward
#                 if getattr(args, "use_rl", False):
#                     from rfdetrv2.rl.reward import compute_batch_reward
#                     rl_reward_type = getattr(args, "rl_reward_type", "combined")
#                     needs_matcher = rl_reward_type in ("hungarian", "hungarian_full")
#                     rl_reward = compute_batch_reward(
#                         outputs, new_targets, postprocess,
#                         reward_type=rl_reward_type,
#                         matcher=getattr(criterion, "matcher", None) if needs_matcher else None,
#                         group_detr=getattr(args, "group_detr", 1) if needs_matcher else 1,
#                     )
#                     rl_weight = getattr(args, "rl_weight", 0.5)
#                     rl_scale = 1.0 + rl_weight * (1.0 - rl_reward)
#                     losses = losses * rl_scale
#                 del outputs

#             scaler.scale(losses).backward()

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         if rl_reward is not None:
#             loss_dict_reduced["rl_reward"] = torch.tensor(rl_reward, device=device, dtype=torch.float32)
#         loss_dict_reduced_unscaled = {
#             f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
#         }
#         loss_dict_reduced_scaled = {
#             k:  v * weight_dict[k]
#             for k, v in loss_dict_reduced.items()
#             if k in weight_dict
#         }
#         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

#         loss_value = losses_reduced_scaled.item()

#         if not math.isfinite(loss_value):
#             print(loss_dict_reduced)
#             raise ValueError("Loss is {}, stopping training".format(loss_value))

#         if max_norm > 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

#         scaler.step(optimizer)
#         scaler.update()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         if ema_m is not None:
#             if epoch >= 0:
#                 ema_m.update(model)
#         metric_logger.update(
#             loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
#         )
#         metric_logger.update(class_error=loss_dict_reduced["class_error"])
#         if "rl_reward" in loss_dict_reduced:
#             metric_logger.update(rl_reward=loss_dict_reduced["rl_reward"].item())
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# def sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt):
#     """Sweep confidence thresholds and compute precision/recall/F1 at each."""
#     num_classes = len(per_class_data)
#     results = []

#     for conf_thresh in conf_thresholds:
#         per_class_precisions = []
#         per_class_recalls = []
#         per_class_f1s = []

#         for k in range(num_classes):
#             data = per_class_data[k]
#             scores = data['scores']
#             matches = data['matches']
#             ignore = data['ignore']
#             total_gt = data['total_gt']

#             above_thresh = scores >= conf_thresh
#             valid = above_thresh & ~ignore

#             valid_matches = matches[valid]

#             tp = np.sum(valid_matches != 0)
#             fp = np.sum(valid_matches == 0)
#             fn = total_gt - tp

#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#             f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

#             per_class_precisions.append(precision)
#             per_class_recalls.append(recall)
#             per_class_f1s.append(f1)

#         if len(classes_with_gt) > 0:
#             macro_precision = np.mean([per_class_precisions[k] for k in classes_with_gt])
#             macro_recall = np.mean([per_class_recalls[k] for k in classes_with_gt])
#             macro_f1 = np.mean([per_class_f1s[k] for k in classes_with_gt])
#         else:
#             macro_precision = 0.0
#             macro_recall = 0.0
#             macro_f1 = 0.0

#         results.append({
#             'confidence_threshold': conf_thresh,
#             'macro_f1': macro_f1,
#             'macro_precision': macro_precision,
#             'macro_recall': macro_recall,
#             'per_class_prec': np.array(per_class_precisions),
#             'per_class_rec': np.array(per_class_recalls),
#         })

#     return results


# def coco_extended_metrics(coco_eval):
#     """
#     Compute precision/recall by sweeping confidence thresholds to maximize macro-F1.
#     Uses evalImgs directly to compute metrics from raw matching data.
#     """

#     iou50_idx = np.argwhere(np.isclose(coco_eval.params.iouThrs, 0.50)).item()
#     cat_ids = coco_eval.params.catIds
#     num_classes = len(cat_ids)
#     area_idx = 0
#     maxdet_idx = 2

#     # Unflatten evalImgs into a nested dict
#     evalImgs_unflat = {}
#     for e in coco_eval.evalImgs:
#         if e is None:
#             continue
#         cat_id = e['category_id']
#         area_rng = tuple(e['aRng'])
#         img_id = e['image_id']

#         if cat_id not in evalImgs_unflat:
#             evalImgs_unflat[cat_id] = {}
#         if area_rng not in evalImgs_unflat[cat_id]:
#             evalImgs_unflat[cat_id][area_rng] = {}
#         evalImgs_unflat[cat_id][area_rng][img_id] = e

#     area_rng_all = tuple(coco_eval.params.areaRng[area_idx])

#     per_class_data = []
#     for cid in cat_ids:
#         dt_scores = []
#         dt_matches = []
#         dt_ignore = []
#         total_gt = 0

#         for img_id in coco_eval.params.imgIds:
#             e = evalImgs_unflat.get(cid, {}).get(area_rng_all, {}).get(img_id)
#             if e is None:
#                 continue

#             num_dt = len(e['dtIds'])
#             # num_gt = len(e['gtIds'])

#             gt_ignore = e['gtIgnore']
#             total_gt += sum(1 for ig in gt_ignore if not ig)

#             for d in range(num_dt):
#                 dt_scores.append(e['dtScores'][d])
#                 dt_matches.append(e['dtMatches'][iou50_idx, d])
#                 dt_ignore.append(e['dtIgnore'][iou50_idx, d])

#         per_class_data.append({
#             'scores': np.array(dt_scores),
#             'matches': np.array(dt_matches),
#             'ignore': np.array(dt_ignore, dtype=bool),
#             'total_gt': total_gt,
#         })

#     conf_thresholds = np.linspace(0.0, 1.0, 101)
#     classes_with_gt = [k for k in range(num_classes) if per_class_data[k]['total_gt'] > 0]

#     confidence_sweep_metric_dicts = sweep_confidence_thresholds(
#         per_class_data, conf_thresholds, classes_with_gt
#     )

#     best = max(confidence_sweep_metric_dicts, key=lambda x: x['macro_f1'])

#     map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

#     per_class = []
#     cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
#     for k, cid in enumerate(cat_ids):

#         # [T, R, K, A, M] -> [T, R]
#         p_slice = coco_eval.eval['precision'][:, :, k, area_idx, maxdet_idx]

#         # [T, R]
#         p_masked = np.where(p_slice > -1, p_slice, np.nan)

#         # We do this as two sequential nanmeans to avoid
#         # underweighting columns with more nans, since each
#         # column corresponds to a different IoU threshold
#         # [T, R] -> [T]
#         ap_per_iou = np.nanmean(p_masked, axis=1)

#         # [T] -> [1]
#         ap_50_95 = float(np.nanmean(ap_per_iou))
#         ap_50    = float(np.nanmean(p_masked[iou50_idx]))

#         if (
#             np.isnan(ap_50_95)
#             or np.isnan(ap_50)
#             or np.isnan(best['per_class_prec'][k])
#             or np.isnan(best['per_class_rec'][k])
#         ):
#             continue

#         per_class.append({
#             "class"      : cat_id_to_name[int(cid)],
#             "map@50:95"  : ap_50_95,
#             "map@50"     : ap_50,
#             "precision"  : best['per_class_prec'][k],
#             "recall"     : best['per_class_rec'][k],
#         })

#     per_class.append({
#         "class"     : "all",
#         "map@50:95" : map_50_95,
#         "map@50"    : map_50,
#         "precision" : best['macro_precision'],
#         "recall"    : best['macro_recall'],
#     })

#     # Build a compact AP-per-class table payload for downstream reporting.
#     per_class_without_all = [row for row in per_class if row["class"] != "all"]
#     per_class_sorted = sorted(per_class_without_all, key=lambda x: x["map@50:95"], reverse=True)
#     ap_per_class_rows = [
#         {
#             "rank": rank + 1,
#             "class": row["class"],
#             "ap50_95": float(row["map@50:95"]),
#             "ap50": float(row["map@50"]),
#         }
#         for rank, row in enumerate(per_class_sorted)
#     ]
#     summary_row = next((row for row in per_class if row["class"] == "all"), None)
#     if summary_row is not None:
#         ap_per_class_rows.append(
#             {
#                 "rank": "all",
#                 "class": "all",
#                 "ap50_95": float(summary_row["map@50:95"]),
#                 "ap50": float(summary_row["map@50"]),
#             }
#         )

#     ap_table_columns = ["rank", "class", "ap50_95", "ap50"]
#     ap_table_lines = [
#         "| rank | class | ap50_95 | ap50 |",
#         "| --- | --- | ---: | ---: |",
#     ]
#     for row in ap_per_class_rows:
#         ap_table_lines.append(
#             f"| {row['rank']} | {row['class']} | {row['ap50_95']:.4f} | {row['ap50']:.4f} |"
#         )
#     ap_per_class_markdown = "\n".join(ap_table_lines)

#     return {
#         "class_map": per_class,
#         "ap_per_class_table": {
#             "columns": ap_table_columns,
#             "rows": ap_per_class_rows,
#         },
#         "ap_per_class_markdown": ap_per_class_markdown,
#         "map"      : map_50,
#         "precision": best['macro_precision'],
#         "recall"   : best['macro_recall'],
#     }

# def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
#     model.eval()
#     if args.fp16_eval:
#         model.half()
#     criterion.eval()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter(
#         "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
#     )
#     header = "Test:"

#     iou_types = ("bbox",) if not args.segmentation_head else ("bbox", "segm")
#     coco_evaluator = CocoEvaluator(base_ds, iou_types, args.eval_max_dets)

#     for samples, targets in metric_logger.log_every(data_loader, 10, header):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         if args.fp16_eval:
#             samples.tensors = samples.tensors.half()

#         # Add autocast for evaluation
#         with autocast(**get_autocast_args(args)):
#             outputs = model(samples)

#         if args.fp16_eval:
#             for key in outputs.keys():
#                 if key == "enc_outputs":
#                     for sub_key in outputs[key].keys():
#                         outputs[key][sub_key] = outputs[key][sub_key].float()
#                 elif key == "aux_outputs":
#                     for idx in range(len(outputs[key])):
#                         for sub_key in outputs[key][idx].keys():
#                             outputs[key][idx][sub_key] = outputs[key][idx][
#                                 sub_key
#                             ].float()
#                 else:
#                     outputs[key] = outputs[key].float()

#         loss_dict = criterion(outputs, targets)
#         weight_dict = criterion.weight_dict

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {
#             k: v * weight_dict[k]
#             for k, v in loss_dict_reduced.items()
#             if k in weight_dict
#         }
#         loss_dict_reduced_unscaled = {
#             f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
#         }
#         metric_logger.update(
#             loss=sum(loss_dict_reduced_scaled.values()),
#             **loss_dict_reduced_scaled,
#             **loss_dict_reduced_unscaled,
#         )
#         metric_logger.update(class_error=loss_dict_reduced["class_error"])

#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results_all = postprocess(outputs, orig_target_sizes)
#         res = {
#             target["image_id"].item(): output
#             for target, output in zip(targets, results_all)
#         }
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     if coco_evaluator is not None:
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     if coco_evaluator is not None:
#         results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
#         stats["results_json"] = results_json
#         if utils.is_main_process():
#             print("\nPer-class AP table (bbox):")
#             print(results_json["ap_per_class_markdown"])
#         if "bbox" in iou_types:
#             stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

#         if "segm" in iou_types:
#             results_json = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
#             if utils.is_main_process():
#                 print("\nPer-class AP table (segm):")
#                 print(results_json["ap_per_class_markdown"])
#             stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
#     return stats, coco_evaluator






# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import random
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F

import rfdetrv2.util.misc as utils
from rfdetrv2.datasets.coco import compute_multi_scale_scales
from rfdetrv2.datasets.coco_eval import CocoEvaluator

try:
    from torch.amp import GradScaler, autocast
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    DEPRECATED_AMP = True
from typing import Callable, DefaultDict, List

import numpy as np

from rfdetrv2.util.misc import NestedTensor


def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}
    else:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}


# def train_one_epoch(
#     model: torch.nn.Module,
#     criterion: torch.nn.Module,
#     lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
#     data_loader: Iterable,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     epoch: int,
#     batch_size: int,
#     max_norm: float = 0,
#     ema_m: torch.nn.Module = None,
#     schedules: dict = {},
#     num_training_steps_per_epoch=None,
#     vit_encoder_num_layers=None,
#     args=None,
#     callbacks: DefaultDict[str, List[Callable]] = None,
#     postprocess=None,
#     # ----------------------------------------------------------------
#     # [GRPO] Thêm tham số mới — None nghĩa là chỉ dùng supervised
#     # ----------------------------------------------------------------
#     grpo_trainer=None,
# ):
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     metric_logger.add_meter(
#         "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
#     )
#     header = "Epoch: [{}]".format(epoch)
#     print_freq = 10
#     start_steps = epoch * num_training_steps_per_epoch
 
#     print("Grad accum steps: ", args.grad_accum_steps)
#     print("Total batch size: ", batch_size * utils.get_world_size())
 
#     if DEPRECATED_AMP:
#         scaler = GradScaler(enabled=args.amp)
#     else:
#         scaler = GradScaler('cuda', enabled=args.amp)
 
#     optimizer.zero_grad()
#     assert batch_size % args.grad_accum_steps == 0
#     sub_batch_size = batch_size // args.grad_accum_steps
 
#     print("LENGTH OF DATA LOADER:", len(data_loader))
 
#     for data_iter_step, (samples, targets) in enumerate(
#         metric_logger.log_every(data_loader, print_freq, header)
#     ):
#         it = start_steps + data_iter_step
#         callback_dict = {"step": it, "model": model, "epoch": epoch}
#         for callback in callbacks["on_train_batch_start"]:
#             callback(callback_dict)
 
#         if "dp" in schedules:
#             if args.distributed:
#                 model.module.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
#             else:
#                 model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
#         if "do" in schedules:
#             if args.distributed:
#                 model.module.update_dropout(schedules["do"][it])
#             else:
#                 model.update_dropout(schedules["do"][it])
 
#         if args.multi_scale and not args.do_random_resize_via_padding:
#             scales = compute_multi_scale_scales(
#                 args.resolution, args.expanded_scales,
#                 args.patch_size, args.num_windows
#             )
#             random.seed(it)
#             scale = random.choice(scales)
#             with torch.no_grad():
#                 samples.tensors = F.interpolate(
#                     samples.tensors, size=scale,
#                     mode='bilinear', align_corners=False
#                 )
#                 samples.mask = F.interpolate(
#                     samples.mask.unsqueeze(1).float(), size=scale, mode='nearest'
#                 ).squeeze(1).bool()
 
#         # ----------------------------------------------------------------
#         # [GRPO] Metrics dict được tích lũy qua grad_accum_steps
#         # ----------------------------------------------------------------
#         accumulated_grpo_metrics: dict = {}
 
#         for i in range(args.grad_accum_steps):
#             start_idx = i * sub_batch_size
#             final_idx = start_idx + sub_batch_size
#             new_samples = NestedTensor(
#                 samples.tensors[start_idx:final_idx],
#                 samples.mask[start_idx:final_idx],
#             ).to(device)
#             new_targets = [
#                 {k: v.to(device) for k, v in t.items()}
#                 for t in targets[start_idx:final_idx]
#             ]
 
#             with autocast(**get_autocast_args(args)):
#                 outputs   = model(new_samples, new_targets)
#                 loss_dict = criterion(outputs, new_targets)
 
#                 # --------------------------------------------------------
#                 # [GRPO] Thay thế toàn bộ block RL cũ
#                 # --------------------------------------------------------
#                 if grpo_trainer is not None:
#                     # GRPOTrainer.step() tính L_sup + λ * L_grpo
#                     losses, grpo_metrics = grpo_trainer.step(
#                         new_samples,
#                         new_targets,
#                         outputs,
#                         loss_dict,
#                         step=it,
#                     )
#                     losses = losses / args.grad_accum_steps
 
#                     # Tích lũy metrics để log sau
#                     for k, v in grpo_metrics.items():
#                         accumulated_grpo_metrics[k] = (
#                             accumulated_grpo_metrics.get(k, 0.0) + v / args.grad_accum_steps
#                         )
 
#                 else:
#                     # ---- Supervised only (không có GRPO) ----
#                     weight_dict = criterion.weight_dict
#                     losses = sum(
#                         (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
#                         for k in loss_dict.keys()
#                         if k in weight_dict
#                     )
#                 # --------------------------------------------------------
 
#                 del outputs
 
#             scaler.scale(losses).backward()
 
#         # ----------------------------------------------------------------
#         # Logging
#         # ----------------------------------------------------------------
#         loss_dict_reduced          = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
#         weight_dict                = criterion.weight_dict
#         loss_dict_reduced_scaled   = {
#             k: v * weight_dict[k]
#             for k, v in loss_dict_reduced.items()
#             if k in weight_dict
#         }
#         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
#         loss_value = losses_reduced_scaled.item()
 
#         if not math.isfinite(loss_value):
#             print(loss_dict_reduced)
#             raise ValueError("Loss is {}, stopping training".format(loss_value))
 
#         if max_norm > 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
 
#         scaler.step(optimizer)
#         scaler.update()
#         lr_scheduler.step()
#         optimizer.zero_grad()
 
#         if ema_m is not None and epoch >= 0:
#             ema_m.update(model)
 
#         metric_logger.update(
#             loss=loss_value,
#             **loss_dict_reduced_scaled,
#             **loss_dict_reduced_unscaled,
#         )
#         metric_logger.update(class_error=loss_dict_reduced["class_error"])
 
#         # [GRPO] Log GRPO metrics nếu có
#         if accumulated_grpo_metrics:
#             metric_logger.update(**accumulated_grpo_metrics)
 
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
 
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
    postprocess=None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch
 
    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())
 
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler('cuda', enabled=args.amp)
 
    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps

    print("LENGTH OF DATA LOADER:", len(data_loader))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        callback_dict = {"step": it, "model": model, "epoch": epoch}
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
 
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])
 
        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(
                args.resolution, args.expanded_scales,
                args.patch_size, args.num_windows,
            )
            random.seed(it)
            scale = random.choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(
                    samples.tensors, size=scale, mode='bilinear', align_corners=False
                )
                samples.mask = F.interpolate(
                    samples.mask.unsqueeze(1).float(), size=scale, mode='nearest'
                ).squeeze(1).bool()

        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples = NestedTensor(
                samples.tensors[start_idx:final_idx],
                samples.mask[start_idx:final_idx],
            ).to(device)
            new_targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets[start_idx:final_idx]
            ]
 
            # ============================================================
            # SUPERVISED PASS — giống hệt engine.py gốc, không thay đổi gì
            # ============================================================
            with autocast(**get_autocast_args(args)):
                outputs   = model(new_samples, new_targets)
                loss_dict = criterion(outputs, new_targets)
                weight_dict = criterion.weight_dict
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )
            del outputs
            scaler.scale(losses).backward()

        # Optimizer step — giống gốc hoàn toàn
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
 
        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError("Loss is {}, stopping training".format(loss_value))
 
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
 
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
 
        if ema_m is not None and epoch >= 0:
            ema_m.update(model)
 
        metric_logger.update(
            loss=loss_value,
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        # lwdetr_query: log prototype alignment loss
        if "loss_proto_align" in loss_dict_reduced:
            metric_logger.update(loss_proto_align=loss_dict_reduced["loss_proto_align"].item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
 
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt):
    """Sweep confidence thresholds and compute precision/recall/F1 at each."""
    num_classes = len(per_class_data)
    results = []

    for conf_thresh in conf_thresholds:
        per_class_precisions = []
        per_class_recalls = []
        per_class_f1s = []

        for k in range(num_classes):
            data = per_class_data[k]
            scores = data['scores']
            matches = data['matches']
            ignore = data['ignore']
            total_gt = data['total_gt']

            above_thresh = scores >= conf_thresh
            valid = above_thresh & ~ignore

            valid_matches = matches[valid]

            tp = np.sum(valid_matches != 0)
            fp = np.sum(valid_matches == 0)
            fn = total_gt - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_precisions.append(precision)
            per_class_recalls.append(recall)
            per_class_f1s.append(f1)

        if len(classes_with_gt) > 0:
            macro_precision = np.mean([per_class_precisions[k] for k in classes_with_gt])
            macro_recall = np.mean([per_class_recalls[k] for k in classes_with_gt])
            macro_f1 = np.mean([per_class_f1s[k] for k in classes_with_gt])
        else:
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0

        results.append({
            'confidence_threshold': conf_thresh,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'per_class_prec': np.array(per_class_precisions),
            'per_class_rec': np.array(per_class_recalls),
        })

    return results


def coco_extended_metrics(coco_eval):
    """
    Compute precision/recall by sweeping confidence thresholds to maximize macro-F1.
    Uses evalImgs directly to compute metrics from raw matching data.
    """

    iou50_idx = np.argwhere(np.isclose(coco_eval.params.iouThrs, 0.50)).item()
    cat_ids = coco_eval.params.catIds
    num_classes = len(cat_ids)
    area_idx = 0
    maxdet_idx = 2

    # Unflatten evalImgs into a nested dict
    evalImgs_unflat = {}
    for e in coco_eval.evalImgs:
        if e is None:
            continue
        cat_id = e['category_id']
        area_rng = tuple(e['aRng'])
        img_id = e['image_id']

        if cat_id not in evalImgs_unflat:
            evalImgs_unflat[cat_id] = {}
        if area_rng not in evalImgs_unflat[cat_id]:
            evalImgs_unflat[cat_id][area_rng] = {}
        evalImgs_unflat[cat_id][area_rng][img_id] = e

    area_rng_all = tuple(coco_eval.params.areaRng[area_idx])

    per_class_data = []
    for cid in cat_ids:
        dt_scores = []
        dt_matches = []
        dt_ignore = []
        total_gt = 0

        for img_id in coco_eval.params.imgIds:
            e = evalImgs_unflat.get(cid, {}).get(area_rng_all, {}).get(img_id)
            if e is None:
                continue

            num_dt = len(e['dtIds'])
            # num_gt = len(e['gtIds'])

            gt_ignore = e['gtIgnore']
            total_gt += sum(1 for ig in gt_ignore if not ig)

            for d in range(num_dt):
                dt_scores.append(e['dtScores'][d])
                dt_matches.append(e['dtMatches'][iou50_idx, d])
                dt_ignore.append(e['dtIgnore'][iou50_idx, d])

        per_class_data.append({
            'scores': np.array(dt_scores),
            'matches': np.array(dt_matches),
            'ignore': np.array(dt_ignore, dtype=bool),
            'total_gt': total_gt,
        })

    conf_thresholds = np.linspace(0.0, 1.0, 101)
    classes_with_gt = [k for k in range(num_classes) if per_class_data[k]['total_gt'] > 0]

    confidence_sweep_metric_dicts = sweep_confidence_thresholds(
        per_class_data, conf_thresholds, classes_with_gt
    )

    best = max(confidence_sweep_metric_dicts, key=lambda x: x['macro_f1'])

    map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    per_class = []
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    for k, cid in enumerate(cat_ids):

        # [T, R, K, A, M] -> [T, R]
        p_slice = coco_eval.eval['precision'][:, :, k, area_idx, maxdet_idx]

        # [T, R]
        p_masked = np.where(p_slice > -1, p_slice, np.nan)

        # We do this as two sequential nanmeans to avoid
        # underweighting columns with more nans, since each
        # column corresponds to a different IoU threshold
        # [T, R] -> [T]
        ap_per_iou = np.nanmean(p_masked, axis=1)

        # [T] -> [1]
        ap_50_95 = float(np.nanmean(ap_per_iou))
        ap_50    = float(np.nanmean(p_masked[iou50_idx]))

        if (
            np.isnan(ap_50_95)
            or np.isnan(ap_50)
            or np.isnan(best['per_class_prec'][k])
            or np.isnan(best['per_class_rec'][k])
        ):
            continue

        per_class.append({
            "class"      : cat_id_to_name[int(cid)],
            "map@50:95"  : ap_50_95,
            "map@50"     : ap_50,
            "precision"  : best['per_class_prec'][k],
            "recall"     : best['per_class_rec'][k],
        })

    per_class.append({
        "class"     : "all",
        "map@50:95" : map_50_95,
        "map@50"    : map_50,
        "precision" : best['macro_precision'],
        "recall"    : best['macro_recall'],
    })

    # Build a compact AP-per-class table payload for downstream reporting.
    per_class_without_all = [row for row in per_class if row["class"] != "all"]
    per_class_sorted = sorted(per_class_without_all, key=lambda x: x["map@50:95"], reverse=True)
    ap_per_class_rows = [
        {
            "rank": rank + 1,
            "class": row["class"],
            "ap50_95": float(row["map@50:95"]),
            "ap50": float(row["map@50"]),
        }
        for rank, row in enumerate(per_class_sorted)
    ]
    summary_row = next((row for row in per_class if row["class"] == "all"), None)
    if summary_row is not None:
        ap_per_class_rows.append(
            {
                "rank": "all",
                "class": "all",
                "ap50_95": float(summary_row["map@50:95"]),
                "ap50": float(summary_row["map@50"]),
            }
        )

    ap_table_columns = ["rank", "class", "ap50_95", "ap50"]
    ap_table_lines = [
        "| rank | class | ap50_95 | ap50 |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in ap_per_class_rows:
        ap_table_lines.append(
            f"| {row['rank']} | {row['class']} | {row['ap50_95']:.4f} | {row['ap50']:.4f} |"
        )
    ap_per_class_markdown = "\n".join(ap_table_lines)

    return {
        "class_map": per_class,
        "ap_per_class_table": {
            "columns": ap_table_columns,
            "rows": ap_per_class_rows,
        },
        "ap_per_class_markdown": ap_per_class_markdown,
        "map"      : map_50,
        "precision": best['macro_precision'],
        "recall"   : best['macro_recall'],
    }

def map_eval_labels_to_coco_category_ids(
    results_all: List[dict],
    label_to_cat_id: Sequence[int],
) -> List[dict]:
    """Map model class indices 0..K-1 to COCO ``category_id`` for pycocotools evaluation."""
    if not label_to_cat_id:
        return results_all
    out: List[dict] = []
    for res in results_all:
        if "labels" not in res:
            out.append(res)
            continue
        labels = res["labels"]
        mapped = torch.tensor(
            [label_to_cat_id[int(x.item())] for x in labels],
            dtype=labels.dtype,
            device=labels.device,
        )
        out.append({**res, "labels": mapped})
    return out


def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = ("bbox",) if not args.segmentation_head else ("bbox", "segm")
    coco_evaluator = CocoEvaluator(base_ds, iou_types, args.eval_max_dets)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][
                                sub_key
                            ].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_all = postprocess(outputs, orig_target_sizes)
        lid = getattr(args, "label_to_cat_id", None)
        if lid is not None:
            results_all = map_eval_labels_to_coco_category_ids(results_all, lid)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results_all)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        stats["results_json"] = results_json
        if utils.is_main_process():
            print("\nPer-class AP table (bbox):")
            print(results_json["ap_per_class_markdown"])
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in iou_types:
            results_json = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
            if utils.is_main_process():
                print("\nPer-class AP table (segm):")
                print(results_json["ap_per_class_markdown"])
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator
