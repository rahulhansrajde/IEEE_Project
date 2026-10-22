"""Deterministic seeding across ``random``, ``numpy``, ``torch`` and cuDNN."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_seed", "seed_worker", "make_generator"]


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every random number generator used by the pipeline.

    Args:
        seed: global seed.
        deterministic: force deterministic cuDNN kernels and warn on
            non-deterministic PyTorch ops.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` giving every worker a distinct, reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
