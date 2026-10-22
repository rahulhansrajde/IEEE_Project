"""Dataset factory.

Supported sources
-----------------
* ``dermamnist`` (default), ``pathmnist``, ``pneumoniamnist``, ``bloodmnist``,
  ``breastmnist``, ``octmnist`` -- downloaded automatically through the official
  ``medmnist`` package (MedMNIST v2 / MedMNIST+, sizes 28, 64, 128, 224). No API
  key is needed.
* ``imagefolder`` -- a local directory, either with ``train/``, ``val/``,
  ``test/`` sub-folders or a flat ``class_name/*.jpg`` layout (stratified split).
* ``ham10000`` -- the raw HAM10000 release (``HAM10000_metadata.csv`` + JPEGs in
  any sub-folder); split by ``lesion_id`` so that images of the same lesion never
  leak across splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.seed import make_generator, seed_worker

from .transforms import build_transforms

__all__ = [
    "MEDMNIST_FLAGS",
    "DERMAMNIST_SHORT_NAMES",
    "LabeledImageDataset",
    "MedMNISTDataset",
    "DatasetBundle",
    "build_datasets",
    "build_dataloaders",
    "stratified_split",
    "get_labels",
]

_LOG = logging.getLogger(__name__)

MEDMNIST_FLAGS = {
    "dermamnist": "DermaMNIST",
    "pathmnist": "PathMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "bloodmnist": "BloodMNIST",
    "breastmnist": "BreastMNIST",
    "octmnist": "OCTMNIST",
    "organamnist": "OrganAMNIST",
}

# HAM10000 / DermaMNIST diagnostic categories in DermaMNIST label order.
DERMAMNIST_SHORT_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class LabeledImageDataset(Dataset):
    """Dataset of image file paths with integer labels."""

    def __init__(self, paths: Sequence[str | Path], labels: Sequence[int], transform: Optional[Callable] = None) -> None:
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have equal length")
        self.paths = [str(p) for p in paths]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


class MedMNISTDataset(Dataset):
    """Thin wrapper around a ``medmnist`` split returning ``(image, int_label)``."""

    def __init__(self, flag: str, split: str, size: int, root: str | Path, transform: Optional[Callable] = None, download: bool = True) -> None:
        import medmnist

        if flag not in MEDMNIST_FLAGS:
            raise ValueError(f"Unsupported MedMNIST flag '{flag}'")
        Path(root).mkdir(parents=True, exist_ok=True)
        cls = getattr(medmnist, MEDMNIST_FLAGS[flag])
        size_kw = {} if size == 28 else {"size": size}
        self._ds = cls(split=split, transform=None, download=download, root=str(root), **size_kw)
        labels = np.asarray(self._ds.labels)
        self.labels = labels[:, 0].astype(np.int64) if labels.ndim > 1 else labels.astype(np.int64)
        self.transform = transform
        info = medmnist.INFO[flag]
        self.class_descriptions = [info["label"][str(i)] for i in range(len(info["label"]))]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self._ds.imgs[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    class_names: List[str]
    num_classes: int
    class_counts: List[int]
    source: str
    extra: Dict[str, Any] = field(default_factory=dict)


def get_labels(ds: Dataset) -> np.ndarray:
    """Integer labels of a dataset (supports nested :class:`Subset`)."""
    if isinstance(ds, Subset):
        return get_labels(ds.dataset)[np.asarray(ds.indices)]
    return np.asarray(ds.labels)


def stratified_split(labels: np.ndarray, fractions: Sequence[float], seed: int = 42, groups: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Split indices into len(fractions) stratified parts (optionally group-aware)."""
    from sklearn.model_selection import StratifiedGroupKFold

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    fractions = np.asarray(fractions, dtype=np.float64)
    fractions = fractions / fractions.sum()
    if groups is None:
        parts: List[List[int]] = [[] for _ in fractions]
        for c in np.unique(labels):
            idx = rng.permutation(np.where(labels == c)[0])
            bounds = np.round(np.cumsum(fractions) * len(idx)).astype(int)
            start = 0
            for i, end in enumerate(bounds):
                parts[i].extend(idx[start:end].tolist())
                start = end
        return [np.sort(np.asarray(p, dtype=np.int64)) for p in parts]

    # Group-aware: assign folds of a StratifiedGroupKFold to the requested parts.
    n_folds = 20
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ids = np.zeros(len(labels), dtype=np.int64)
    for fold, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        fold_ids[test_idx] = fold
    fold_bounds = np.round(np.cumsum(fractions) * n_folds).astype(int)
    out, start = [], 0
    for end in fold_bounds:
        out.append(np.where((fold_ids >= start) & (fold_ids < end))[0])
        start = end
    return out


def _limit(ds: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    """Stratified sub-sample of at most ``max_samples`` items (for smoke tests)."""
    if max_samples is None or len(ds) <= max_samples:
        return ds
    labels = get_labels(ds)
    frac = max_samples / len(ds)
    idx = stratified_split(labels, [frac, 1 - frac], seed=seed)[0]
    if len(idx) == 0:
        idx = np.arange(max_samples)
    return Subset(ds, idx.tolist())


def _scan_class_folders(root: Path) -> Tuple[List[Path], List[int], List[str]]:
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    paths, labels = [], []
    for ci, cname in enumerate(classes):
        for p in sorted((root / cname).rglob("*")):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
                labels.append(ci)
    return paths, labels, classes


def _build_imagefolder(data_dir: Path, transforms: Dict[str, Callable], fractions: Sequence[float], seed: int):
    split_dirs = {s: data_dir / s for s in ("train", "val", "test")}
    if all(d.is_dir() for d in split_dirs.values()):
        _, _, classes = _scan_class_folders(split_dirs["train"])
        datasets = {}
        for split, d in split_dirs.items():
            paths, labels, cls = _scan_class_folders(d)
            if cls != classes:
                mapping = {c: classes.index(c) for c in cls}
                labels = [mapping[cls[l]] for l in labels]
            datasets[split] = LabeledImageDataset(paths, labels, transforms[split])
        return datasets, classes
    paths, labels, classes = _scan_class_folders(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")
    labels_arr = np.asarray(labels)
    parts = stratified_split(labels_arr, fractions, seed=seed)
    datasets = {}
    for split, idx in zip(("train", "val", "test"), parts):
        datasets[split] = LabeledImageDataset([paths[i] for i in idx], labels_arr[idx], transforms[split])
    return datasets, classes


def _build_ham10000(data_dir: Path, transforms: Dict[str, Callable], fractions: Sequence[float], seed: int):
    import pandas as pd

    meta_files = list(data_dir.rglob("HAM10000_metadata*.csv"))
    if not meta_files:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found under {data_dir}")
    meta = pd.read_csv(meta_files[0])
    image_index = {p.stem: p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    meta = meta[meta["image_id"].isin(image_index)].reset_index(drop=True)
    if meta.empty:
        raise FileNotFoundError("No HAM10000 images matching the metadata were found")
    classes = DERMAMNIST_SHORT_NAMES
    labels = meta["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    if np.isnan(labels.astype(np.float64)).any():
        raise ValueError("Unknown diagnosis code in HAM10000 metadata")
    labels = labels.astype(np.int64)
    groups = meta["lesion_id"].to_numpy()
    parts = stratified_split(labels, fractions, seed=seed, groups=groups)
    paths = [image_index[i] for i in meta["image_id"]]
    datasets = {}
    for split, idx in zip(("train", "val", "test"), parts):
        datasets[split] = LabeledImageDataset([paths[i] for i in idx], labels[idx], transforms[split])
    return datasets, classes


def build_datasets(data_cfg: Dict[str, Any], seed: int = 42) -> DatasetBundle:
    """Build train/val/test datasets from the ``data`` section of a config."""
    source = str(data_cfg.get("dataset", "dermamnist")).lower()
    image_size = int(data_cfg.get("image_size", 224))
    native_size = int(data_cfg.get("native_size", image_size if image_size in (28, 64, 128, 224) else 224))
    clahe = bool(data_cfg.get("clahe", False))
    augment = bool(data_cfg.get("augment", True))
    transforms = {
        "train": build_transforms(image_size, train=True, augment=augment, clahe=clahe),
        "val": build_transforms(image_size, train=False, clahe=clahe),
        "test": build_transforms(image_size, train=False, clahe=clahe),
    }
    fractions = data_cfg.get("split_fractions", [0.7, 0.1, 0.2])
    root = Path(data_cfg.get("root", "data"))

    if source in MEDMNIST_FLAGS:
        datasets = {
            split: MedMNISTDataset(source, split, native_size, root, transforms[split], download=data_cfg.get("download", True))
            for split in ("train", "val", "test")
        }
        n_cls = len(datasets["train"].class_descriptions)
        if source == "dermamnist":
            class_names = list(DERMAMNIST_SHORT_NAMES)
        else:
            class_names = [d.split(" ")[0][:12] if len(d) > 14 else d for d in datasets["train"].class_descriptions]
        extra = {"class_descriptions": datasets["train"].class_descriptions, "native_size": native_size}
    elif source == "imagefolder":
        if not data_cfg.get("data_dir"):
            raise ValueError("data.data_dir must be set for the imagefolder source")
        datasets, class_names = _build_imagefolder(Path(data_cfg["data_dir"]), transforms, fractions, seed)
        n_cls = len(class_names)
        extra = {"data_dir": str(data_cfg["data_dir"])}
    elif source == "ham10000":
        if not data_cfg.get("data_dir"):
            raise ValueError("data.data_dir must be set for the ham10000 source")
        datasets, class_names = _build_ham10000(Path(data_cfg["data_dir"]), transforms, fractions, seed)
        n_cls = len(class_names)
        extra = {"data_dir": str(data_cfg["data_dir"])}
    else:
        raise ValueError(f"Unknown dataset source '{source}'")

    limits = data_cfg.get("max_samples_per_split") or {}
    if isinstance(limits, int):
        limits = {"train": limits, "val": limits, "test": limits}
    for split in ("train", "val", "test"):
        datasets[split] = _limit(datasets[split], limits.get(split), seed)

    train_labels = get_labels(datasets["train"])
    counts = np.bincount(train_labels, minlength=n_cls).tolist()
    return DatasetBundle(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        class_names=class_names,
        num_classes=n_cls,
        class_counts=counts,
        source=source,
        extra=extra,
    )


def build_dataloaders(data_cfg: Dict[str, Any], seed: int = 42) -> Tuple[DatasetBundle, Dict[str, DataLoader]]:
    """Datasets plus seeded ``DataLoader`` objects for every split."""
    bundle = build_datasets(data_cfg, seed)
    batch_size = int(data_cfg.get("batch_size", 64))
    eval_batch_size = int(data_cfg.get("eval_batch_size", batch_size * 2))
    num_workers = int(data_cfg.get("num_workers", 2))
    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )
    loaders = {
        "train": DataLoader(bundle.train, batch_size=batch_size, shuffle=True, drop_last=len(bundle.train) > batch_size,
                            generator=make_generator(seed), **common),
        "val": DataLoader(bundle.val, batch_size=eval_batch_size, shuffle=False, **common),
        "test": DataLoader(bundle.test, batch_size=eval_batch_size, shuffle=False, **common),
    }
    return bundle, loaders
