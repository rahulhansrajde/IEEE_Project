"""Rich console + file logging and a JSON metric tracker."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["get_logger", "MetricTracker", "to_serialisable"]

try:
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:  # rich is optional at runtime
    _HAS_RICH = False


def get_logger(name: str = "attconvkan", log_file: Optional[str | Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to a rich console handler and, optionally, a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(h, "_attconvkan_console", False) for h in logger.handlers):
        if _HAS_RICH:
            console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
            console.setFormatter(logging.Formatter("%(message)s"))
        else:
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        console._attconvkan_console = True
        logger.addHandler(console)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_file.resolve())
        if not any(getattr(h, "baseFilename", None) == resolved for h in logger.handlers):
            fh = logging.FileHandler(resolved, mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(fh)
    return logger


def to_serialisable(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars and arrays into JSON-safe types."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if isinstance(obj, dict):
        return {str(k): to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serialisable(v) for v in obj]
    if torch is not None and isinstance(obj, torch.Tensor):
        return to_serialisable(obj.detach().cpu().tolist())
    if np is not None and isinstance(obj, np.ndarray):
        return to_serialisable(obj.tolist())
    if np is not None and isinstance(obj, np.generic):
        return to_serialisable(obj.item())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


class MetricTracker:
    """Accumulates per-epoch metric dictionaries and persists them as JSON."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path is not None else None
        self.history: List[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]) -> None:
        self.history.append(to_serialisable(record))
        if self.path is not None:
            self.save()

    def best(self, key: str, mode: str = "max") -> Optional[Dict[str, Any]]:
        rows = [r for r in self.history if r.get(key) is not None]
        if not rows:
            return None
        return max(rows, key=lambda r: r[key]) if mode == "max" else min(rows, key=lambda r: r[key])

    def save(self, path: Optional[str | Path] = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("No path given for MetricTracker.save")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.history, indent=2))
        return target

    @staticmethod
    def save_json(obj: Any, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_serialisable(obj), indent=2))
        return path
