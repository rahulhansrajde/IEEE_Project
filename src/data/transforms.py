"""Clinical image augmentations, CLAHE contrast enhancement and Mixup/CutMix."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "CLAHE", "ToRGB", "build_transforms", "denormalize", "MixupCutmix"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CLAHE:
    """Contrast-Limited Adaptive Histogram Equalisation on the luminance channel.

    Operates in CIE-Lab space so that the dermoscopic colour information is
    preserved while local contrast of lesion borders is enhanced.
    """

    def __init__(self, clip_limit: float = 0.01, kernel_size: Optional[int] = None) -> None:
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    def __call__(self, img: Image.Image) -> Image.Image:
        from skimage import color, exposure

        rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        lab = color.rgb2lab(rgb)
        lum = lab[..., 0] / 100.0
        lum = exposure.equalize_adapthist(lum, kernel_size=self.kernel_size, clip_limit=self.clip_limit)
        lab[..., 0] = lum * 100.0
        out = np.clip(color.lab2rgb(lab), 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(clip_limit={self.clip_limit})"


class ToRGB:
    """Convert any PIL image (grayscale, RGBA, ...) to 3-channel RGB."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def build_transforms(
    image_size: int = 224,
    train: bool = True,
    augment: bool = True,
    clahe: bool = False,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """Torchvision pipeline for dermoscopy / histology / radiography images.

    Training augmentations follow standard dermoscopy practice: random resized
    crops, horizontal/vertical flips, rotations (lesions have no canonical
    orientation) and mild colour jitter to model inter-device variability.
    """
    ops = [ToRGB()]
    if clahe:
        ops.append(CLAHE())
    if train and augment:
        ops += [
            T.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(30)], p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        ]
    else:
        ops.append(T.Resize((image_size, image_size), antialias=True))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def denormalize(
    x: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Invert :func:`torchvision.transforms.Normalize` and clamp to [0, 1]."""
    m = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    s = torch.tensor(std, device=x.device).view(-1, 1, 1)
    return (x * s + m).clamp(0.0, 1.0)


class MixupCutmix:
    """Batch-level Mixup (Zhang et al., 2018) / CutMix (Yun et al., 2019).

    Calling the object returns ``(images, targets_a, targets_b, lam)``; the loss
    is ``lam * L(y_a) + (1 - lam) * L(y_b)``.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        switch_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator) -> Tuple[int, int, int, int]:
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
        cy, cx = rng.integers(h), rng.integers(w)
        y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
        x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
        return int(y1), int(y2), int(x1), int(x2)

    def __call__(self, images: torch.Tensor, targets: torch.Tensor):
        if self.rng.random() > self.prob:
            return images, targets, targets, 1.0
        perm = torch.from_numpy(self.rng.permutation(images.size(0))).to(images.device)
        targets_b = targets[perm]
        use_cutmix = self.cutmix_alpha > 0 and (self.mixup_alpha <= 0 or self.rng.random() < self.switch_prob)
        if use_cutmix:
            lam = float(self.rng.beta(self.cutmix_alpha, self.cutmix_alpha))
            y1, y2, x1, x2 = self._rand_bbox(images.size(2), images.size(3), lam, self.rng)
            images = images.clone()
            images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (images.size(2) * images.size(3)))
        else:
            lam = float(self.rng.beta(self.mixup_alpha, self.mixup_alpha))
            images = lam * images + (1.0 - lam) * images[perm]
        return images, targets, targets_b, float(lam)
