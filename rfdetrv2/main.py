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
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
cleaned main file
"""
import argparse
import ast
import copy
import datetime
import json
import math
import multiprocessing
import os
import random
import shutil
import time
import warnings
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Callable, DefaultDict, List

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler

import rfdetrv2.util.misc as utils
from rfdetrv2.datasets import build_dataset, get_coco_api_from_dataset
from rfdetrv2.engine import evaluate, train_one_epoch
from rfdetrv2.models import PostProcess, build_criterion_and_postprocessors, build_model
from rfdetrv2.util.benchmark import benchmark
from rfdetrv2.util.drop_scheduler import drop_scheduler
from rfdetrv2.util.files import download_file
from rfdetrv2.util.get_param_dicts import get_param_dict
from rfdetrv2.util.utils import BestMetricHolder, ModelEma, clean_state_dict

if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

logger = getLogger(__name__)


def download_pretrain_weights(pretrain_weights: str, redownload=False):
    if pretrain_weights in HOSTED_MODELS:
        if redownload or not os.path.exists(pretrain_weights):
            logger.info(f"Downloading pretrained weights for {pretrain_weights}")
            download_file(HOSTED_MODELS[pretrain_weights], pretrain_weights)


class Model:
    def __init__(self, **kwargs):
        args = populate_args(**kwargs)
        self.args = args
        self.resolution = args.resolution
        self.model = build_model(args)
        self.device = torch.device(args.device)

        if args.pretrain_weights is not None:
            print("Loading pretrain weights")
            try:
                checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Failed to load pretrain weights: {e}")
                print("Failed to load pretrain weights, re-downloading")
                download_pretrain_weights(args.pretrain_weights, redownload=True)
                checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)

            if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
                self.args.class_names = checkpoint['args'].class_names
                self.class_names = checkpoint['args'].class_names

            checkpoint_num_classes = checkpoint['model']['class_embed.bias'].shape[0]
            if checkpoint_num_classes != args.num_classes + 1:
                self.reinitialize_detection_head(checkpoint_num_classes)

            if args.pretrain_exclude_keys is not None:
                assert isinstance(args.pretrain_exclude_keys, list)
                for exclude_key in args.pretrain_exclude_keys:
                    checkpoint['model'].pop(exclude_key)

            if args.pretrain_keys_modify_to_load is not None:
                from rfdetrv2.util.obj365_to_coco_model import get_coco_pretrain_from_obj365
                assert isinstance(args.pretrain_keys_modify_to_load, list)
                for modify_key_to_load in args.pretrain_keys_modify_to_load:
                    try:
                        checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                            model_without_ddp.state_dict()[modify_key_to_load],
                            checkpoint['model'][modify_key_to_load]
                        )
                    except:
                        print(f"Failed to load {modify_key_to_load}, deleting from checkpoint")
                        checkpoint['model'].pop(modify_key_to_load)

            num_desired_queries = args.num_queries * args.group_detr
            query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
            for name, state in checkpoint['model'].items():
                if any(name.endswith(x) for x in query_param_names):
                    checkpoint['model'][name] = state[:num_desired_queries]

            self.model.load_state_dict(checkpoint['model'], strict=False)

        if args.backbone_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as exc:
                raise ImportError(
                    "backbone_lora=True requires `peft` and compatible `transformers`/`torch` versions. "
                    "Please install compatible packages or disable backbone_lora."
                ) from exc
            print("Applying LORA to backbone")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                use_dora=True,
                target_modules=[
                    "q_proj", "v_proj", "k_proj",
                    "qkv",
                    "query", "key", "value", "cls_token", "register_tokens",
                ]
            )
            self.model.backbone[0].encoder = get_peft_model(self.model.backbone[0].encoder, lora_config)

        self.model = self.model.to(self.device)
        self.postprocess = PostProcess(num_select=args.num_select)
        self.stop_early = False

    def reinitialize_detection_head(self, num_classes):
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self):
        self.stop_early = True
        print("Early stopping requested, will complete current epoch and stop")

    def train(self, callbacks: DefaultDict[str, List[Callable]], **kwargs):
        currently_supported_callbacks = ["on_fit_epoch_end", "on_train_batch_start", "on_train_end"]
        for key in callbacks.keys():
            if key not in currently_supported_callbacks:
                raise ValueError(
                    f"Callback {key} is not currently supported, please file an issue if you need it!\n"
                    f"Currently supported callbacks: {currently_supported_callbacks}"
                )

        args = populate_args(**kwargs)
        if getattr(args, 'class_names') is not None:
            self.args.class_names = args.class_names
            self.args.num_classes = args.num_classes

        utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(utils.get_sha()))
        print(args)
        device = torch.device(args.device)

        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        criterion, postprocess = build_criterion_and_postprocessors(args)

        # ----------------------------------------------------------------
        # Freeze SwiftNet backbone when freeze_encoder=True
        # ----------------------------------------------------------------
        if getattr(args, 'freeze_encoder', False):
            backbone = self.model.backbone
            if hasattr(backbone, '__getitem__') and len(backbone) > 0 and hasattr(backbone[0], 'encoder'):
                for p in backbone[0].encoder.parameters():
                    p.requires_grad = False
                if utils.is_main_process():
                    logger.info("SwiftNet encoder frozen (no gradient update).")

        model = self.model
        model.to(device)

        model_without_ddp = model
        if args.distributed:
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=False
            )
            model_without_ddp = model.module

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        world_size = utils.get_world_size()
        if world_size > 1:
            lr_scale_mode = getattr(args, 'lr_scale_mode', 'sqrt')
            if lr_scale_mode == 'sqrt':
                lr_scale = math.sqrt(world_size)
            else:
                lr_scale = float(world_size)
            args.lr = args.lr * lr_scale
            args.lr_encoder = args.lr_encoder * lr_scale
            if utils.is_main_process():
                print(
                    f'Multi-GPU: scaling LR by {lr_scale_mode}(world_size={world_size}) '
                    f'-> lr={args.lr:.2e}, lr_encoder={args.lr_encoder:.2e}'
                )

        param_dicts = get_param_dict(args, model_without_ddp)
        param_dicts = [p for p in param_dicts if p['params'].requires_grad]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

        dataset_train = build_dataset(image_set='train', args=args, resolution=args.resolution)
        dataset_val   = build_dataset(image_set='val',   args=args, resolution=args.resolution)
        dataset_test  = build_dataset(
            image_set='test' if args.dataset_file == "roboflow" else "val",
            args=args, resolution=args.resolution,
        )

        debug_data_limit = int(getattr(args, "debug_data_limit", 0) or 0)
        if debug_data_limit > 0:
            train_n = min(debug_data_limit, len(dataset_train))
            val_n   = min(debug_data_limit, len(dataset_val))
            test_n  = min(debug_data_limit, len(dataset_test))
            dataset_train = torch.utils.data.Subset(dataset_train, range(train_n))
            dataset_val   = torch.utils.data.Subset(dataset_val,   range(val_n))
            dataset_test  = torch.utils.data.Subset(dataset_test,  range(test_n))
            logger.info(
                "Debug data limit enabled: train=%d, val=%d, test=%d",
                train_n, val_n, test_n,
            )

        total_batch_size_for_lr = args.batch_size * utils.get_world_size() * args.grad_accum_steps
        num_training_steps_per_epoch_lr = (
            len(dataset_train) + total_batch_size_for_lr - 1
        ) // total_batch_size_for_lr
        total_training_steps_lr = num_training_steps_per_epoch_lr * args.epochs
        warmup_steps_lr         = num_training_steps_per_epoch_lr * args.warmup_epochs
        lr_gamma      = getattr(args, 'lr_gamma', 0.1)
        lr_milestones = getattr(args, 'lr_milestones', [80, 160])
        milestone_steps = [m * num_training_steps_per_epoch_lr for m in sorted(lr_milestones)]
        if args.lr_scheduler == 'multistep' and utils.is_main_process():
            print(f'LR scheduler: multistep, milestones={lr_milestones} epochs, gamma={lr_gamma}')

        def lr_lambda(current_step: int):
            # Phase 1: warmup
            if current_step < warmup_steps_lr:
                return float(current_step) / float(max(1, warmup_steps_lr))

            progress = float(current_step - warmup_steps_lr) / float(
                max(1, total_training_steps_lr - warmup_steps_lr)
            )

            if args.lr_scheduler == 'cosine':
                return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

            if args.lr_scheduler == 'wsd':
                # Warmup-Stable-Decay: stable phase then cosine decay
                stable_ratio = getattr(args, 'lr_stable_ratio', 0.7)
                if progress < stable_ratio:
                    return 1.0
                decay_progress = (progress - stable_ratio) / max(1e-8, 1.0 - stable_ratio)
                return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (
                    1 + math.cos(math.pi * decay_progress)
                )

            if args.lr_scheduler == 'cosine_restart':
                # Cosine với warm restart mỗi restart_period epochs
                # Giúp thoát local minima khi plateau (epoch 20+)
                restart_period = getattr(args, 'lr_restart_period', 50)
                restart_decay  = getattr(args, 'lr_restart_decay', 0.8)
                restart_steps  = int(restart_period * num_training_steps_per_epoch_lr)
                elapsed        = current_step - warmup_steps_lr
                cycle_idx      = elapsed // max(1, restart_steps)
                cycle_progress = (elapsed % max(1, restart_steps)) / max(1, restart_steps)
                # LR peak giảm dần qua các cycle
                peak_factor    = restart_decay ** cycle_idx
                return peak_factor * (
                    args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (
                        1 + math.cos(math.pi * cycle_progress)
                    )
                )

            if args.lr_scheduler == 'linear':
                return 1.0 - progress * (1.0 - args.lr_min_factor)

            if args.lr_scheduler == 'step':
                drop_step = args.lr_drop * num_training_steps_per_epoch_lr
                return 1.0 if current_step < drop_step else lr_gamma

            if args.lr_scheduler == 'multistep':
                gamma_exp = sum(1 for ms in milestone_steps if current_step >= ms)
                return lr_gamma ** gamma_exp

            return 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val   = DistributedSampler(dataset_val,  shuffle=False)
            sampler_test  = DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test  = torch.utils.data.SequentialSampler(dataset_test)

        effective_batch_size = args.batch_size * args.grad_accum_steps
        if debug_data_limit > 0:
            min_batches = kwargs.get('min_batches', 1)
        else:
            min_batches = kwargs.get('min_batches', 5)

        num_workers = args.num_workers
        if num_workers > 0 and multiprocessing.get_start_method(allow_none=True) == 'spawn':
            import __main__
            if not hasattr(__main__, '__file__') or not __main__.__name__ == '__main__':
                warnings.warn(
                    "Setting num_workers to 0 because the script is not wrapped in "
                    "`if __name__ == '__main__':`. This is required for multiprocessing "
                    "with the 'spawn' start method.",
                    RuntimeWarning,
                )
                num_workers = 0

        if len(dataset_train) < effective_batch_size * min_batches:
            logger.info(
                f"Training with uniform sampler because dataset is too small: "
                f"{len(dataset_train)} < {effective_batch_size * min_batches}"
            )
            sampler = torch.utils.data.RandomSampler(
                dataset_train,
                replacement=True,
                num_samples=effective_batch_size * min_batches,
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_size=effective_batch_size,
                collate_fn=utils.collate_fn,
                num_workers=num_workers,
                sampler=sampler,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )
        else:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, effective_batch_size, drop_last=True
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=utils.collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )

        data_loader_val = DataLoader(
            dataset_val, args.batch_size, sampler=sampler_val,
            drop_last=False, collate_fn=utils.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        data_loader_test = DataLoader(
            dataset_test, args.batch_size, sampler=sampler_test,
            drop_last=False, collate_fn=utils.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        base_ds      = get_coco_api_from_dataset(dataset_val)
        base_ds_test = get_coco_api_from_dataset(dataset_test)

        if args.use_ema:
            self.ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)
        else:
            self.ema_m = None

        output_dir = Path(args.output_dir)

        if utils.is_main_process():
            print("Get benchmark")
            if args.do_benchmark:
                benchmark_model = copy.deepcopy(model_without_ddp)
                bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
                print(json.dumps(bm, indent=2))
                del benchmark_model

        # ----------------------------------------------------------------
        # Resume từ checkpoint
        # ----------------------------------------------------------------
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

            if args.use_ema:
                if 'ema_model' in checkpoint:
                    self.ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
                else:
                    del self.ema_m
                    self.ema_m = ModelEma(model, decay=args.ema_decay, tau=args.ema_tau)

            if (
                not args.eval
                and 'optimizer' in checkpoint
                and 'lr_scheduler' in checkpoint
                and 'epoch' in checkpoint
            ):
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
        # ----------------------------------------------------------------

        if args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocess, data_loader_val, base_ds, device, args
            )
            if args.output_dir:
                if not args.segmentation_head:
                    utils.save_on_master(
                        coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
                    )
                else:
                    utils.save_on_master(
                        coco_evaluator.coco_eval["segm"].eval, output_dir / "eval.pth"
                    )
            return

        total_batch_size = effective_batch_size * utils.get_world_size()
        num_training_steps_per_epoch = (
            len(dataset_train) + total_batch_size - 1
        ) // total_batch_size

        schedules = {}
        if args.dropout > 0:
            schedules['do'] = drop_scheduler(
                args.dropout, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule,
            )
            print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))

        if args.drop_path > 0:
            schedules['dp'] = drop_scheduler(
                args.drop_path, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule,
            )
            print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))

        print("Start training")
        start_time = time.time()
        best_map_holder   = BestMetricHolder(use_ema=args.use_ema)
        best_map_5095     = 0
        best_map_50       = 0
        best_map_ema_5095 = 0
        best_map_ema_50   = 0

        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()

            if args.distributed:
                sampler_train.set_epoch(epoch)

            model.train()
            criterion.train()

            train_stats = train_one_epoch(
                model, criterion, lr_scheduler, data_loader_train,
                optimizer, device, epoch,
                effective_batch_size, args.clip_max_norm,
                ema_m=self.ema_m,
                schedules=schedules,
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers,
                args=args,
                callbacks=callbacks,
                postprocess=postprocess,
            )

            train_epoch_time     = time.time() - epoch_start_time
            train_epoch_time_str = str(datetime.timedelta(seconds=int(train_epoch_time)))

            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model':        model_without_ddp.state_dict(),
                        'optimizer':    optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch':        epoch,
                        'args':         args,
                    }
                    if args.use_ema:
                        weights['ema_model'] = self.ema_m.module.state_dict()

                    if not args.dont_save_weights:
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        utils.save_on_master(weights, checkpoint_path)

            with torch.no_grad():
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocess, data_loader_val, base_ds, device, args=args
                )

            if not args.segmentation_head:
                map_regular = test_stats["coco_eval_bbox"][0]
            else:
                map_regular = test_stats["coco_eval_masks"][0]

            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                best_map_5095 = max(best_map_5095, map_regular)
                map50 = (
                    test_stats["coco_eval_bbox"][1]
                    if not args.segmentation_head
                    else test_stats["coco_eval_masks"][1]
                )
                best_map_50 = max(best_map_50, map50)
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                if not args.dont_save_weights:
                    best_weights = {
                        'model':        model_without_ddp.state_dict(),
                        'optimizer':    optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch':        epoch,
                        'args':         args,
                    }
                    utils.save_on_master(best_weights, checkpoint_path)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v  for k, v in test_stats.items()},
                'epoch':        epoch,
                'n_parameters': n_parameters,
            }

            if args.use_ema:
                ema_test_stats, _ = evaluate(
                    self.ema_m.module, criterion, postprocess,
                    data_loader_val, base_ds, device, args=args,
                )
                log_stats.update({f'ema_test_{k}': v for k, v in ema_test_stats.items()})
                map_ema = (
                    ema_test_stats["coco_eval_bbox"][0]
                    if not args.segmentation_head
                    else ema_test_stats["coco_eval_masks"][0]
                )
                best_map_ema_5095 = max(best_map_ema_5095, map_ema)
                _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
                if _isbest:
                    map_ema_50 = (
                        ema_test_stats["coco_eval_bbox"][1]
                        if not args.segmentation_head
                        else ema_test_stats["coco_eval_masks"][1]
                    )
                    best_map_ema_50 = max(best_map_ema_50, map_ema_50)
                    checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                    if not args.dont_save_weights:
                        ema_weights = {
                            'model':        self.ema_m.module.state_dict(),
                            'optimizer':    optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch':        epoch,
                            'args':         args,
                        }
                        utils.save_on_master(ema_weights, checkpoint_path)

            log_stats.update(best_map_holder.summary())

            ep_paras = {'epoch': epoch, 'n_parameters': n_parameters}
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            log_stats['train_epoch_time'] = train_epoch_time_str
            epoch_time     = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            if not args.segmentation_head:
                                torch.save(
                                    coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name,
                                )
                            else:
                                torch.save(
                                    coco_evaluator.coco_eval["segm"].eval,
                                    output_dir / "eval" / name,
                                )

            for callback in callbacks["on_fit_epoch_end"]:
                callback(log_stats)

            if self.stop_early:
                print(f"Early stopping requested, stopping at epoch {epoch}")
                break

        best_is_ema = best_map_ema_5095 > best_map_5095

        if utils.is_main_process():
            if best_is_ema:
                shutil.copy2(
                    output_dir / 'checkpoint_best_ema.pth',
                    output_dir / 'checkpoint_best_total.pth',
                )
            else:
                shutil.copy2(
                    output_dir / 'checkpoint_best_regular.pth',
                    output_dir / 'checkpoint_best_total.pth',
                )

            utils.strip_checkpoint(output_dir / 'checkpoint_best_total.pth')

            best_map_5095 = max(best_map_5095, best_map_ema_5095)
            results = ema_test_stats["results_json"] if best_is_ema else test_stats["results_json"]
            class_map = results["class_map"]
            results["class_map"] = {"valid": class_map}
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

            total_time     = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            print('Results saved to {}'.format(output_dir / "results.json"))

        if best_is_ema:
            self.model = self.ema_m.module
        self.model.eval()

        if args.run_test:
            best_state_dict = torch.load(
                output_dir / 'checkpoint_best_total.pth',
                map_location='cpu', weights_only=False,
            )['model']
            model.load_state_dict(best_state_dict)
            model.eval()
            test_stats, _ = evaluate(
                model, criterion, postprocess, data_loader_test, base_ds_test, device, args=args
            )
            print(f"Test results: {test_stats}")
            with open(output_dir / "results.json", "r") as f:
                results = json.load(f)
            test_metrics = test_stats["results_json"]["class_map"]
            results["class_map"]["test"] = test_metrics
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

        for callback in callbacks["on_train_end"]:
            callback()

    def export(
        self,
        output_dir="output",
        infer_dir=None,
        simplify=False,
        backbone_only=False,
        opset_version=17,
        verbose=True,
        force=False,
        shape=None,
        batch_size=1,
        **kwargs,
    ):
        """Export the trained model to ONNX format"""
        print("Exporting model to ONNX format")
        try:
            from rfdetr.deploy.export import export_onnx, make_infer_image, onnx_simplify
        except ImportError:
            print(
                "It seems some dependencies for ONNX export are missing. "
                "Please run `pip install rfdetr[onnxexport]` and try again."
            )
            raise

        device = self.device
        model  = deepcopy(self.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)

        if shape is None:
            shape = (self.resolution, self.resolution)
        else:
            if shape[0] % 14 != 0 or shape[1] % 14 != 0:
                raise ValueError("Shape must be divisible by 14")

        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
        input_names   = ['input']
        output_names  = ['features'] if backbone_only else ['dets', 'labels']
        dynamic_axes  = None

        self.model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                print(f"PyTorch inference output shape: {features.shape}")
            elif self.args.segmentation_head:
                outputs = model(input_tensors)
                print(
                    f"PyTorch inference output shapes — "
                    f"Boxes: {outputs['pred_boxes'].shape}, "
                    f"Labels: {outputs['pred_logits'].shape}, "
                    f"Masks: {outputs['pred_masks'].shape}"
                )
            else:
                outputs = model(input_tensors)
                print(
                    f"PyTorch inference output shapes — "
                    f"Boxes: {outputs['pred_boxes'].shape}, "
                    f"Labels: {outputs['pred_logits'].shape}"
                )

        model.cpu()
        input_tensors = input_tensors.cpu()

        output_file = export_onnx(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version,
        )
        print(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            sim_output_file = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensors,
                force=force,
            )
            print(f"Successfully simplified ONNX model to: {sim_output_file}")

        print("ONNX export completed successfully")
        self.model = self.model.to(device)


# ============================================================
# get_args_parser
# ============================================================

def get_args_parser():
    parser = argparse.ArgumentParser('LWDETR training and evaluation script', add_help=False)

    parser.add_argument('--num_classes',      default=2,    type=int)
    parser.add_argument('--grad_accum_steps', default=1,    type=int)
    parser.add_argument('--amp',              default=False, type=bool)
    parser.add_argument('--lr',               default=2e-4, type=float,
                        help='Decoder LR. Encoder = lr/4 (chuẩn fine-tune ViT).')
    parser.add_argument('--lr_encoder',       default=2.5e-5, type=float,
                        help='Encoder LR. 2.5e-5 → ~7.07e-5 sau sqrt scaling (8 GPU), tỉ lệ 1:8 vs decoder.')
    parser.add_argument('--lr_scale_mode',    default='sqrt', type=str,
                        choices=['linear', 'sqrt'],
                        help='Multi-GPU LR scaling: sqrt (recommended for transformers) or linear.')
    parser.add_argument('--batch_size',       default=2,    type=int)
    parser.add_argument('--weight_decay',     default=1e-4, type=float)
    parser.add_argument('--epochs',           default=12,   type=int)
    parser.add_argument('--lr_drop',          default=11,   type=int)
    parser.add_argument('--clip_max_norm',    default=0.1,  type=float)
    parser.add_argument('--lr_vit_layer_decay', default=0.8,  type=float)
    parser.add_argument('--lr_component_decay', default=1.0,  type=float)
    parser.add_argument('--do_benchmark', action='store_true')

    # Drop args
    parser.add_argument('--dropout',       type=float, default=0)
    parser.add_argument('--drop_path',     type=float, default=0)
    parser.add_argument('--drop_mode',     type=str,   default='standard',
                        choices=['standard', 'early', 'late'])
    parser.add_argument('--drop_schedule', type=str,   default='constant',
                        choices=['constant', 'linear'])
    parser.add_argument('--cutoff_epoch',  type=int,   default=0)

    # Model parameters
    parser.add_argument('--pretrained_encoder',          type=str,  default=None)
    parser.add_argument('--pretrain_weights',            type=str,  default=None)
    parser.add_argument('--pretrain_exclude_keys',       type=str,  default=None, nargs='+')
    parser.add_argument('--pretrain_keys_modify_to_load', type=str, default=None, nargs='+')

    # Backbone
    parser.add_argument('--encoder',               default='vit_tiny', type=str)
    parser.add_argument('--vit_encoder_num_layers', default=12,         type=int)
    parser.add_argument('--window_block_indexes',  default=None,        type=int, nargs='+')
    parser.add_argument('--position_embedding',    default='sine',      type=str,
                        choices=('sine', 'learned'))
    parser.add_argument('--out_feature_indexes',   default=[-1],        type=int, nargs='+')
    parser.add_argument('--freeze_encoder',  action='store_true', dest='freeze_encoder')
    parser.add_argument('--layer_norm',      action='store_true', dest='layer_norm')
    parser.add_argument('--rms_norm',        action='store_true', dest='rms_norm')
    parser.add_argument('--backbone_lora',   action='store_true', dest='backbone_lora')
    parser.add_argument('--force_no_pretrain', action='store_true', dest='force_no_pretrain')

    # Transformer
    parser.add_argument('--dec_layers',          default=3,    type=int)
    parser.add_argument('--dim_feedforward',     default=2048, type=int)
    parser.add_argument('--hidden_dim',          default=256,  type=int)
    parser.add_argument('--sa_nheads',           default=8,    type=int)
    parser.add_argument('--ca_nheads',           default=8,    type=int)
    parser.add_argument('--num_queries',         default=300,  type=int)
    parser.add_argument('--group_detr',          default=13,   type=int)
    parser.add_argument('--two_stage',           action='store_true')
    parser.add_argument('--projector_scale',     default=['P3', 'P4', 'P5'], type=str, nargs='+',
                        choices=('P3', 'P4', 'P5', 'P6'),
                        help='Multi-scale levels. P3=2x upsample, P4=1x, P5=0.5x downsample. Default: P3 P4 P5')
    parser.add_argument('--lite_refpoint_refine', action='store_true')
    parser.add_argument('--num_select',          default=100,  type=int)
    parser.add_argument('--dec_n_points',        default=8,    type=int)
    parser.add_argument('--decoder_norm',        default='LN', type=str)
    parser.add_argument('--bbox_reparam',        action='store_true')
    parser.add_argument('--freeze_batch_norm',   action='store_true')

    # Matcher
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox',  default=5, type=float)
    parser.add_argument('--set_cost_giou',  default=2, type=float)

    # Loss coefficients
    parser.add_argument('--cls_loss_coef',  default=1.0,  type=float)
    parser.add_argument('--bbox_loss_coef', default=5.0,   type=float)
    parser.add_argument('--giou_loss_coef', default=2.0,  type=float)
    parser.add_argument('--focal_alpha',    default=0.25, type=float)
    parser.add_argument('--no_aux_loss',    dest='aux_loss', action='store_false')
    parser.add_argument('--sum_group_losses',             action='store_true')
    parser.add_argument('--use_varifocal_loss',           action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss',                  action='store_true')

    parser.add_argument('--no_convnext_projector',        action='store_true', dest='no_use_convnext_projector')

    # Dataset
    parser.add_argument('--dataset_file',        default="coco")
    parser.add_argument('--coco_path',           type=str)
    parser.add_argument('--dataset_dir',         type=str)
    parser.add_argument('--square_resize_div_64', action='store_true')

    # Output
    parser.add_argument('--output_dir',          default='output')
    parser.add_argument('--dont_save_weights',   action='store_true')
    parser.add_argument('--checkpoint_interval', default=10,  type=int)
    parser.add_argument('--seed',                default=42,  type=int)
    parser.add_argument('--resume',              default='')
    parser.add_argument('--start_epoch',         default=0,   type=int, metavar='N')
    parser.add_argument('--eval',                action='store_true')
    parser.add_argument('--use_ema',             action='store_true')
    parser.add_argument('--ema_decay',           default=0.9997, type=float)
    parser.add_argument('--ema_tau',             default=0,      type=float)
    parser.add_argument('--num_workers',         default=2,      type=int)

    # Distributed
    parser.add_argument('--device',    default='cuda')
    parser.add_argument('--world_size', default=1,       type=int)
    parser.add_argument('--dist_url',   default='env://')
    parser.add_argument('--sync_bn',    default=True,    type=bool)

    # FP16
    parser.add_argument('--fp16_eval', default=False, action='store_true')

    # Custom
    parser.add_argument('--encoder_only',              action='store_true')
    parser.add_argument('--backbone_only',             action='store_true')
    parser.add_argument('--resolution',                type=int, default=640)
    parser.add_argument('--use_cls_token',             action='store_true')
    parser.add_argument('--multi_scale',               action='store_true')
    parser.add_argument('--expanded_scales',           action='store_true')
    parser.add_argument('--do_random_resize_via_padding', action='store_true')
    parser.add_argument('--warmup_epochs', default=1, type=float)
    parser.add_argument('--lr_scheduler', default='linear',
                        choices=['step', 'cosine', 'linear', 'multistep', 'wsd', 'cosine_restart'])
    parser.add_argument('--lr_milestones', default=[80, 160], type=int, nargs='+')
    parser.add_argument('--lr_gamma',      default=0.1,   type=float)
    parser.add_argument('--lr_min_factor', default=0.05,  type=float,
                        help='LR tối thiểu cuối cycle. 0.05 = cosine decay sâu hơn (0.1→jump lớn khi restart).')
    parser.add_argument('--lr_stable_ratio', default=0.7, type=float,
        help='WSD: tỷ lệ tổng steps giữ LR ổn định trước khi cosine decay. '
             '0.7 = LR peak đến 70%% training rồi mới decay.')
    parser.add_argument('--lr_restart_period', default=50, type=int,
        help='Cosine restart: số epochs mỗi cycle.')
    parser.add_argument('--lr_restart_decay', default=0.8, type=float,
        help='Cosine restart: LR peak nhân với decay sau mỗi cycle. '
             '0.8 = cycle2 peak = 80%% cycle1 peak.')

    # Early stopping
    parser.add_argument('--early_stopping',            action='store_true')
    parser.add_argument('--early_stopping_patience',   default=10,    type=int)
    parser.add_argument('--early_stopping_min_delta',  default=0.001, type=float)
    parser.add_argument('--early_stopping_use_ema',    action='store_true')
    parser.add_argument('--debug_data_limit',          default=0,     type=int)

    # Subparsers
    subparsers = parser.add_subparsers(
        title='sub-commands', dest='subcommand',
        description='valid subcommands', help='additional help',
    )
    parser_export = subparsers.add_parser('export_model', help='LWDETR model export')
    parser_export.add_argument('--infer_dir',      type=str,  default=None)
    parser_export.add_argument('--verbose',        type=ast.literal_eval, default=False, nargs="?", const=True)
    parser_export.add_argument('--opset_version',  type=int,  default=17)
    parser_export.add_argument('--simplify',       action='store_true')
    parser_export.add_argument('--tensorrt', '--trtexec', '--trt', action='store_true')
    parser_export.add_argument('--dry-run', '--test', '-t', action='store_true')
    parser_export.add_argument('--profile',        action='store_true')
    parser_export.add_argument('--shape',          type=int,  nargs=2, default=(640, 640))

    return parser


# ============================================================
# populate_args
# ============================================================

def populate_args(
    # Basic training
    num_classes=2,
    grad_accum_steps=1,
    amp=False,
    lr=2e-4,
    lr_encoder=2.5e-5,
    lr_scale_mode='sqrt',
    batch_size=2,
    weight_decay=1e-4,
    epochs=12,
    lr_drop=11,
    clip_max_norm=0.1,
    lr_vit_layer_decay=0.8,
    lr_component_decay=1.0,
    do_benchmark=False,
    # Drop
    dropout=0,
    drop_path=0,
    drop_mode='standard',
    drop_schedule='constant',
    cutoff_epoch=0,
    # Model
    pretrained_encoder=None,
    pretrain_weights=None,
    pretrain_exclude_keys=None,
    pretrain_keys_modify_to_load=None,
    pretrained_distiller=None,
    # Backbone
    encoder='vit_tiny',
    vit_encoder_num_layers=12,
    window_block_indexes=None,
    position_embedding='sine',
    out_feature_indexes=[-1],
    freeze_encoder=False,
    layer_norm=False,
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,
    # Transformer
    dec_layers=3,
    dim_feedforward=2048,
    hidden_dim=256,
    sa_nheads=8,
    ca_nheads=8,
    num_queries=300,
    group_detr=13,
    two_stage=False,
    projector_scale=['P3', 'P4', 'P5'],
    lite_refpoint_refine=False,
    num_select=100,
    dec_n_points=8,
    decoder_norm='LN',
    bbox_reparam=False,
    freeze_batch_norm=False,
    # Matcher
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,
    # Loss
    cls_loss_coef=1.0,
    bbox_loss_coef=5.0,
    giou_loss_coef=2.0,
    focal_alpha=0.25,
    aux_loss=True,
    sum_group_losses=False,
    use_varifocal_loss=False,
    use_position_supervised_loss=False,
    ia_bce_loss=False,
    # Dataset
    dataset_file="coco",
    coco_path=None,
    dataset_dir=None,
    square_resize_div_64=False,
    # Output
    output_dir='output',
    dont_save_weights=False,
    checkpoint_interval=10,
    seed=42,
    resume='',
    start_epoch=0,
    eval=False,
    use_ema=False,
    ema_decay=0.9997,
    ema_tau=0,
    num_workers=2,
    # Distributed
    device='cuda',
    world_size=1,
    dist_url='env://',
    sync_bn=True,
    # FP16
    fp16_eval=False,
    # Custom
    encoder_only=False,
    backbone_only=False,
    resolution=640,
    use_cls_token=False,
    multi_scale=False,
    expanded_scales=False,
    do_random_resize_via_padding=False,
    warmup_epochs=1,
    lr_scheduler='step',
    lr_min_factor=0.05,
    lr_stable_ratio=0.7,
    lr_restart_period=50,
    lr_restart_decay=0.8,
    # Early stopping
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=False,
    debug_data_limit=0,
    gradient_checkpointing=False,
    use_windowed_attn=False,
    use_convnext_projector=True,
    # Additional
    subcommand=None,
    **extra_kwargs,
):
    # Convert no_use_convnext_projector từ get_args_parser
    if extra_kwargs.pop('no_use_convnext_projector', None) is True:
        use_convnext_projector = False

    args = argparse.Namespace(
        num_classes=num_classes,
        grad_accum_steps=grad_accum_steps,
        amp=amp,
        lr=lr,
        lr_encoder=lr_encoder,
        lr_scale_mode=lr_scale_mode,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=epochs,
        lr_drop=lr_drop,
        clip_max_norm=clip_max_norm,
        lr_vit_layer_decay=lr_vit_layer_decay,
        lr_component_decay=lr_component_decay,
        do_benchmark=do_benchmark,
        dropout=dropout,
        drop_path=drop_path,
        drop_mode=drop_mode,
        drop_schedule=drop_schedule,
        cutoff_epoch=cutoff_epoch,
        pretrained_encoder=pretrained_encoder,
        pretrain_weights=pretrain_weights,
        pretrain_exclude_keys=pretrain_exclude_keys,
        pretrain_keys_modify_to_load=pretrain_keys_modify_to_load,
        pretrained_distiller=pretrained_distiller,
        encoder=encoder,
        vit_encoder_num_layers=vit_encoder_num_layers,
        window_block_indexes=window_block_indexes,
        position_embedding=position_embedding,
        out_feature_indexes=out_feature_indexes,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        force_no_pretrain=force_no_pretrain,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        hidden_dim=hidden_dim,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        num_queries=num_queries,
        group_detr=group_detr,
        two_stage=two_stage,
        projector_scale=projector_scale,
        lite_refpoint_refine=lite_refpoint_refine,
        num_select=num_select,
        dec_n_points=dec_n_points,
        decoder_norm=decoder_norm,
        bbox_reparam=bbox_reparam,
        freeze_batch_norm=freeze_batch_norm,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        cls_loss_coef=cls_loss_coef,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        focal_alpha=focal_alpha,
        aux_loss=aux_loss,
        sum_group_losses=sum_group_losses,
        use_varifocal_loss=use_varifocal_loss,
        use_position_supervised_loss=use_position_supervised_loss,
        ia_bce_loss=ia_bce_loss,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        square_resize_div_64=square_resize_div_64,
        output_dir=output_dir,
        dont_save_weights=dont_save_weights,
        checkpoint_interval=checkpoint_interval,
        seed=seed,
        resume=resume,
        start_epoch=start_epoch,
        eval=eval,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        num_workers=num_workers,
        device=device,
        world_size=world_size,
        dist_url=dist_url,
        sync_bn=sync_bn,
        fp16_eval=fp16_eval,
        encoder_only=encoder_only,
        backbone_only=backbone_only,
        resolution=resolution,
        use_cls_token=use_cls_token,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        do_random_resize_via_padding=do_random_resize_via_padding,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min_factor=lr_min_factor,
        lr_stable_ratio=lr_stable_ratio,
        lr_restart_period=lr_restart_period,
        lr_restart_decay=lr_restart_decay,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=early_stopping_use_ema,
        debug_data_limit=debug_data_limit,
        gradient_checkpointing=gradient_checkpointing,
        use_windowed_attn=use_windowed_attn,
        use_convnext_projector=use_convnext_projector,
        subcommand=subcommand,
        **extra_kwargs,
    )
    return args


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'LWDETR training and evaluation script',
        parents=[get_args_parser()],
    )
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = vars(args)

    if args.subcommand == 'distill':
        distill(**config)
    elif args.subcommand is None:
        main(**config)
    elif args.subcommand == 'export_model':
        filter_keys = [
            "num_classes", "grad_accum_steps", "lr", "lr_encoder", "weight_decay",
            "epochs", "lr_drop", "clip_max_norm", "lr_vit_layer_decay",
            "lr_component_decay", "dropout", "drop_path", "drop_mode", "drop_schedule",
            "cutoff_epoch", "pretrained_encoder", "pretrain_weights",
            "pretrain_exclude_keys", "pretrain_keys_modify_to_load",
            "freeze_florence", "freeze_aimv2", "decoder_norm",
            "set_cost_class", "set_cost_bbox", "set_cost_giou",
            "cls_loss_coef", "bbox_loss_coef", "giou_loss_coef", "focal_alpha",
            "aux_loss", "sum_group_losses", "use_varifocal_loss",
            "use_position_supervised_loss", "ia_bce_loss",
            "dataset_file", "coco_path", "dataset_dir", "square_resize_div_64",
            "output_dir", "checkpoint_interval", "seed", "resume", "start_epoch",
            "eval", "use_ema", "ema_decay", "ema_tau", "num_workers", "device",
            "world_size", "dist_url", "sync_bn", "fp16_eval", "infer_dir",
            "verbose", "opset_version", "dry_run", "shape",
        ]
        for key in filter_keys:
            config.pop(key, None)
        from deploy.export import main as export_main
        if args.batch_size != 1:
            config['batch_size'] = 1
        export_main(**config)