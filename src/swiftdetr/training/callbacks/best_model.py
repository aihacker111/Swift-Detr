# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Best-model tracking and early stopping (pure PyTorch, no Lightning)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import torch

from swiftdetr.util.logger import get_logger
from swiftdetr.util.package import get_version
from swiftdetr.util.state_dict import strip_checkpoint

logger = get_logger()


class BestModelTracker:
    """Tracks best validation mAP and saves stripped ``.pth`` checkpoints.

    Args:
        output_dir: Directory where checkpoints are written.
        use_ema: Whether to also track and save EMA checkpoints.
    """

    def __init__(self, output_dir: str, use_ema: bool = False) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._use_ema = use_ema
        self._best_regular: float = 0.0
        self._best_ema: float = 0.0

    @staticmethod
    def _build_payload(
        model_state_dict: dict,
        args_dict: object,
        epoch: int,
        global_step: int,
        model_name: Optional[str] = None,
    ) -> dict:
        try:
            from pytorch_lightning import __version__ as ptl_version  # type: ignore[import-not-found]
        except ImportError:
            ptl_version = "none"

        from swiftdetr.util.state_dict import _make_fit_loop_state

        payload: dict = {
            "model": model_state_dict,
            "args": args_dict,
            "epoch": epoch,
            "state_dict": {f"model.{k}": v for k, v in model_state_dict.items()},
            "global_step": global_step,
            "pytorch-lightning_version": ptl_version,
            "loops": {"fit_loop": _make_fit_loop_state(epoch)},
            "optimizer_states": [],
            "lr_schedulers": [],
        }
        if model_name is not None:
            payload["model_name"] = model_name
        version = get_version()
        if version is not None:
            payload["swiftdetr_version"] = version
        return payload

    def update(
        self,
        metrics: dict,
        model_state_dict: dict,
        args_dict: object,
        epoch: int,
        global_step: int,
        ema_state_dict: Optional[dict] = None,
        model_name: Optional[str] = None,
        class_names: Optional[list] = None,
    ) -> None:
        """Save best regular and EMA checkpoints when mAP improves."""
        map_val = metrics.get("mAP_50_95", 0.0)
        if map_val > self._best_regular:
            self._best_regular = map_val
            path = self._output_dir / "checkpoint_best_regular.pth"
            torch.save(
                self._build_payload(model_state_dict, args_dict, epoch, global_step, model_name),
                path,
            )
            logger.info("Best regular mAP=%.4f saved → %s (epoch %d)", map_val, path, epoch)

        if self._use_ema and ema_state_dict is not None:
            ema_map = metrics.get("ema_mAP_50_95", 0.0)
            if ema_map > self._best_ema:
                self._best_ema = ema_map
                path = self._output_dir / "checkpoint_best_ema.pth"
                torch.save(
                    self._build_payload(ema_state_dict, args_dict, epoch, global_step, model_name),
                    path,
                )
                logger.info("Best EMA mAP=%.4f saved → %s (epoch %d)", ema_map, path, epoch)

    def finalize(self) -> None:
        """Select the overall best (regular vs EMA) and copy to ``checkpoint_best_total.pth``."""
        regular_path = self._output_dir / "checkpoint_best_regular.pth"
        ema_path = self._output_dir / "checkpoint_best_ema.pth"
        total_path = self._output_dir / "checkpoint_best_total.pth"

        best_is_ema = self._best_ema > self._best_regular
        best_path = ema_path if (best_is_ema and ema_path.exists()) else regular_path

        if best_path and best_path.exists():
            shutil.copy2(best_path, total_path)
            strip_checkpoint(total_path)
            logger.info(
                "Best total checkpoint from %s (regular=%.4f ema=%.4f) → %s",
                "EMA" if best_is_ema else "regular",
                self._best_regular,
                self._best_ema,
                total_path,
            )


class EarlyStoppingTracker:
    """Patience-based early stopping for validation mAP.

    Args:
        patience: Max epochs without improvement.
        min_delta: Minimum improvement to reset the counter.
        use_ema: Monitor EMA metric only when ``True``; otherwise ``max(regular, ema)``.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001, use_ema: bool = False) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._use_ema = use_ema
        self._best: float = 0.0
        self._wait: int = 0
        self.should_stop: bool = False

    def update(self, metrics: dict) -> None:
        regular = metrics.get("mAP_50_95", None)
        ema = metrics.get("ema_mAP_50_95", None)

        if regular is None and ema is None:
            return

        if self._use_ema and ema is not None:
            effective = ema
        elif regular is not None and ema is not None:
            effective = max(regular, ema)
        elif ema is not None:
            effective = ema
        else:
            effective = regular  # type: ignore[assignment]

        if effective > self._best + self._min_delta:
            self._best = effective
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                logger.info(
                    "Early stopping triggered: no improvement for %d epochs (best=%.4f).",
                    self._patience,
                    self._best,
                )
                self.should_stop = True
