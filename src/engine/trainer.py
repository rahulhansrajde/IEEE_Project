"""Training engine: AMP, warm-up + CosineAnnealingWarmRestarts, early stopping, KAN grid updates."""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.transforms import MixupCutmix
from src.engine.evaluator import compute_metrics, predict
from src.losses.composite_loss import CompositeKANLoss
from src.models.baselines import get_new_module_names
from src.models.kan_layers import KANLinear

__all__ = ["EarlyStopping", "build_optimizer", "Trainer", "resolve_device"]


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopping:
    """Stop when the monitored metric has not improved for ``patience`` epochs."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.counter = 0

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """Update with a new value; return ``True`` if training should stop."""
        if self.is_improvement(value):
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_optimizer(model: nn.Module, train_cfg: Dict[str, Any], model_name: Optional[str] = None) -> torch.optim.Optimizer:
    """AdamW with separate learning rates for pretrained and newly initialised modules.

    Norm layers, biases and 1-D parameters are excluded from weight decay.
    """
    lr = float(train_cfg.get("lr", 5e-4))
    head_mult = float(train_cfg.get("head_lr_multiplier", 1.0))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    new_prefixes = tuple(f"{n}." for n in get_new_module_names(model, model_name))
    groups: Dict[str, Dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new = name.startswith(new_prefixes)
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        key = f"{'new' if is_new else 'pre'}_{'nd' if no_decay else 'd'}"
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr * (head_mult if is_new else 1.0),
                "weight_decay": 0.0 if no_decay else wd,
                "name": key,
            }
        groups[key]["params"].append(param)
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999))
    if opt_name == "sgd":
        return torch.optim.SGD(list(groups.values()), momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer '{opt_name}'")


class Trainer:
    """Supervised trainer with validation-driven model selection.

    Args:
        model: network to train.
        loss_fn: :class:`CompositeKANLoss` instance.
        train_cfg: ``training`` section of the experiment config.
        device: compute device.
        checkpoint_path: where the best checkpoint is written.
        logger: logger used for progress messages.
        model_name: registry name (used for parameter grouping).
        num_classes: number of classes for validation metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: CompositeKANLoss,
        train_cfg: Dict[str, Any],
        device: torch.device,
        checkpoint_path: Optional[str | Path] = None,
        logger: Optional[logging.Logger] = None,
        model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.cfg = train_cfg
        self.device = device
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.logger = logger or logging.getLogger("attconvkan")
        self.model_name = model_name
        self.num_classes = num_classes

        self.epochs = int(train_cfg.get("epochs", 50))
        self.warmup_epochs = float(train_cfg.get("warmup_epochs", 1))
        self.grad_clip = float(train_cfg.get("grad_clip", 0.0) or 0.0)
        self.monitor = str(train_cfg.get("monitor", "val_macro_f1"))
        self.monitor_mode = "min" if "loss" in self.monitor else "max"
        self.grid_update_epochs = set(int(e) for e in train_cfg.get("grid_update_epochs", []) or [])
        self.grid_update_samples = int(train_cfg.get("grid_update_samples", 512))
        self.max_train_batches = train_cfg.get("max_train_batches")
        self.use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

        self.optimizer = build_optimizer(self.model, train_cfg, model_name)
        sched_cfg = train_cfg.get("scheduler", {}) or {}
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(1, int(sched_cfg.get("T_0", max(1, self.epochs - int(self.warmup_epochs))))),
            T_mult=int(sched_cfg.get("T_mult", 1)),
            eta_min=float(sched_cfg.get("eta_min", 1e-6)),
        )
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.early_stopping = EarlyStopping(
            patience=int(train_cfg.get("early_stopping_patience", 10)), mode=self.monitor_mode
        )
        mix_cfg = train_cfg.get("mixup", {}) or {}
        self.mixer = (
            MixupCutmix(
                mixup_alpha=float(mix_cfg.get("mixup_alpha", 0.2)),
                cutmix_alpha=float(mix_cfg.get("cutmix_alpha", 1.0)),
                prob=float(mix_cfg.get("prob", 0.5)),
                seed=int(train_cfg.get("seed", 0)),
            )
            if mix_cfg.get("enabled", False)
            else None
        )
        self.history: List[Dict[str, Any]] = []
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch = -1

    # ----------------------------------------------------------- schedule
    def _set_lr(self, epoch_float: float) -> None:
        if epoch_float < self.warmup_epochs:
            factor = max(epoch_float / max(self.warmup_epochs, 1e-8), 1e-3)
            for g, base in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = base * factor
        else:
            self.scheduler.step(epoch_float - self.warmup_epochs)

    # ----------------------------------------------------------- grid update
    @torch.no_grad()
    def update_kan_grids(self, loader: DataLoader) -> int:
        """Refit every :class:`KANLinear` grid to the current input distribution."""
        layers = [m for m in self.model.modules() if isinstance(m, KANLinear)]
        if not layers:
            return 0
        captured: Dict[int, List[torch.Tensor]] = {id(l): [] for l in layers}
        handles = [
            l.register_forward_pre_hook(lambda mod, inp: captured[id(mod)].append(inp[0].detach().reshape(-1, mod.in_features)))
            for l in layers
        ]
        self.model.eval()
        n_seen = 0
        try:
            for images, _ in loader:
                self.model(images.to(self.device))
                n_seen += images.size(0)
                if n_seen >= self.grid_update_samples:
                    break
        finally:
            for h in handles:
                h.remove()
        for layer in layers:
            x = torch.cat(captured[id(layer)])[: self.grid_update_samples]
            layer.update_grid(x)
        self.model.train()
        return len(layers)

    # ----------------------------------------------------------- epochs
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        n_batches = len(loader)
        if self.max_train_batches:
            n_batches = min(n_batches, int(self.max_train_batches))
        totals = {"loss": 0.0, "cls_loss": 0.0, "spline_l1": 0.0, "spline_entropy": 0.0}
        n_samples, n_correct = 0, 0
        for i, (images, labels) in enumerate(loader):
            if i >= n_batches:
                break
            self._set_lr(epoch + i / n_batches)
            images = images.to(self.device, non_blocking=True)
            labels = torch.as_tensor(labels).view(-1).to(self.device, non_blocking=True)
            targets_b, lam = None, 1.0
            if self.mixer is not None:
                images, labels, targets_b, lam = self.mixer(images, labels)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                out = self.loss_fn(logits, labels, self.model, targets_b=targets_b, lam=lam)
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {i}")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            bs = labels.size(0)
            n_samples += bs
            n_correct += int((logits.argmax(1) == labels).sum())
            for k in totals:
                totals[k] += float(out[k].detach()) * bs
        result = {f"train_{k}": v / max(n_samples, 1) for k, v in totals.items()}
        result["train_accuracy"] = n_correct / max(n_samples, 1)
        return result

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        out = predict(self.model, loader, self.device, amp=self.use_amp)
        logits = torch.from_numpy(out["logits"]).to(self.device)
        labels = torch.from_numpy(out["labels"]).to(self.device)
        loss = float(self.loss_fn.focal(logits, labels))
        n_cls = self.num_classes or out["probs"].shape[1]
        metrics = compute_metrics(out["labels"], out["probs"], n_cls)
        return {"val_loss": loss, **{f"val_{k}": v for k, v in metrics.items()}}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> List[Dict[str, Any]]:
        """Train for ``epochs`` epochs (or until early stopping) and keep the best weights."""
        start_time = time.time()
        for epoch in range(self.epochs):
            if epoch in self.grid_update_epochs:
                n = self.update_kan_grids(train_loader)
                if n:
                    self.logger.info("  epoch %d: updated B-spline grids of %d KANLinear layers", epoch + 1, n)
            t0 = time.time()
            record: Dict[str, Any] = {"epoch": epoch + 1}
            record.update(self.train_one_epoch(train_loader, epoch))
            record.update(self.validate(val_loader))
            record["lr"] = max(g["lr"] for g in self.optimizer.param_groups)
            record["epoch_time_s"] = time.time() - t0
            self.history.append(record)
            value = record[self.monitor]
            improved = self.early_stopping.is_improvement(value)
            if improved:
                self.best_epoch = epoch + 1
                self.best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
                self.save_checkpoint(record)
            self.logger.info(
                "  epoch %02d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_f1=%.4f  val_auc=%.4f  (%.1fs)%s",
                epoch + 1, self.epochs, record["train_loss"], record["val_loss"], record["val_accuracy"],
                record["val_macro_f1"], record["val_auroc"], record["epoch_time_s"], "  *best*" if improved else "",
            )
            if self.early_stopping.step(value):
                self.logger.info("  early stopping at epoch %d (best epoch %d)", epoch + 1, self.best_epoch)
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.total_time_s = time.time() - start_time
        return self.history

    def save_checkpoint(self, record: Dict[str, Any]) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "epoch": record["epoch"],
            "metrics": {k: v for k, v in record.items() if isinstance(v, (int, float)) and not math.isnan(v)},
            "model_name": self.model_name,
        }
        torch.save(payload, self.checkpoint_path)

    def history_frame(self):
        import pandas as pd

        return pd.DataFrame(self.history)
