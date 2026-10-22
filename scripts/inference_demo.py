#!/usr/bin/env python
"""Single-image diagnostic demo for a trained Att-ConvKAN checkpoint.

Prints the class probabilities and writes
  * ``<out>/<stem>_gradcam.png|pdf``  -- input, attention map and Grad-CAM++ overlay,
  * ``<out>/<stem>_splines.png|pdf``  -- the learned KAN edge functions that
    contribute most to the predicted class, with this image's input marked.

Examples::

    python scripts/inference_demo.py --image lesion.jpg
    python scripts/inference_demo.py --sample-index 12      # DermaMNIST test image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_datasets, build_transforms, denormalize  # noqa: E402
from src.engine import resolve_device  # noqa: E402
from src.explainability import GradCAMPlusPlus, overlay_heatmap, plot_edge_responses, save_figure, set_ieee_style  # noqa: E402
from src.models import PROPOSED_MODEL, build_model, get_gradcam_target_layer  # noqa: E402
from src.data.datasets import DERMAMNIST_SHORT_NAMES  # noqa: E402

import run_all_experiments  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an RGB image")
    src.add_argument("--sample-index", type=int, help="index into the configured test split")
    p.add_argument("--config", default=str(ROOT / "configs" / "quick_benchmark.yaml"))
    p.add_argument("--checkpoint", default=None, help="defaults to <checkpoint_dir>/att_convkan_proposed_best.pt")
    p.add_argument("--model", default=PROPOSED_MODEL)
    p.add_argument("--class-names", nargs="*", default=None)
    p.add_argument("--output-dir", default=str(ROOT / "results" / "inference"))
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = run_all_experiments.load_config(args.config)
    data_root = Path(cfg["data"].get("root", "data"))
    cfg["data"]["root"] = str(data_root if data_root.is_absolute() else ROOT / data_root)
    device = resolve_device(args.device)
    image_size = int(cfg["data"]["image_size"])

    label = None
    if args.sample_index is not None:
        bundle = build_datasets(cfg["data"])
        x, label = bundle.test[args.sample_index]
        class_names = bundle.class_names
        stem = f"test_{args.sample_index}"
    else:
        tf = build_transforms(image_size, train=False, clahe=bool(cfg["data"].get("clahe", False)))
        with Image.open(args.image) as img:
            x = tf(img.convert("RGB"))
        class_names = args.class_names or DERMAMNIST_SHORT_NAMES
        stem = Path(args.image).stem

    ckpt_dir = Path(cfg["experiment"].get("checkpoint_dir", "checkpoints"))
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (ckpt_dir if ckpt_dir.is_absolute() else ROOT / ckpt_dir) / f"{args.model}_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint {ckpt_path} not found -- run scripts/run_all_experiments.py first.")
    model = build_model(args.model, len(class_names), pretrained=False, image_size=image_size, model_cfg=cfg.get("model", {}))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"])
    model = model.to(device).eval()

    xb = x.unsqueeze(0).to(device)
    with GradCAMPlusPlus(model, get_gradcam_target_layer(model)) as cam:
        cams, logits = cam(xb)
    probs = torch.softmax(logits, 1)[0].cpu().numpy()
    pred = int(probs.argmax())

    print(f"\nDiagnosis probabilities for {stem}:")
    for i in np.argsort(probs)[::-1]:
        bar = "#" * int(round(40 * probs[i]))
        print(f"  {class_names[i]:>8s}  {100 * probs[i]:6.2f}%  {bar}")
    print(f"\nPrediction: {class_names[pred]}" + (f"   (ground truth: {class_names[label]})" if label is not None else ""))

    out_dir = Path(args.output_dir)
    import matplotlib.pyplot as plt

    set_ieee_style()
    img = denormalize(x).permute(1, 2, 0).numpy()
    panels = [(img, "Input")]
    attention = getattr(model, "attention", None)
    if attention is not None and attention.last_attention_map is not None:
        with torch.no_grad():
            model(xb)
        att = F.interpolate(attention.last_attention_map.unsqueeze(1), size=xb.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0].cpu().numpy()
        att = (att - att.min()) / (np.ptp(att) + 1e-8)
        panels.append((overlay_heatmap(img, att, 0.5, "viridis"), "Coordinate-channel attention"))
    panels.append((overlay_heatmap(img, cams[0]), f"Grad-CAM++ ({class_names[pred]}, {100 * probs[pred]:.1f}%)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.4))
    for ax, (im, title) in zip(np.atleast_1d(axes), panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    paths = save_figure(fig, out_dir, f"{stem}_gradcam")

    head_kan = getattr(getattr(model, "head", None), "kan2", None)
    if head_kan is not None:
        captured = []
        h = head_kan.register_forward_pre_hook(lambda m, inp: captured.append(inp[0].detach()))
        with torch.no_grad():
            model(xb)
        h.remove()
        paths += plot_edge_responses(head_kan, captured[0][0], pred, out_dir, f"{stem}_splines", k=8,
                                     out_name=class_names[pred])
    print("\nSaved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
