#!/usr/bin/env python
"""Master orchestrator for the Att-ConvKAN study.

For every model of Table I (proposed + 5 baselines) and Table II (4 ablations):
train with an identical recipe, select the best validation epoch, evaluate on
the held-out test split (clinical metrics, bootstrap CIs, ECE, split-conformal
coverage), profile hardware efficiency, then generate all publication figures
(Grad-CAM++, learned splines, ROC/PR, confusion matrices, Pareto, training
dynamics, t-SNE) and the LaTeX tables.

Per-model results are written to ``results/metrics/<model>_*.{json,csv}`` so runs
are resumable (finished models are skipped unless ``--force``) and can be split
across processes (``--models``) and merged afterwards (``--aggregate-only``).

Example::

    python scripts/run_all_experiments.py --config configs/quick_benchmark.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders, denormalize  # noqa: E402
from src.engine import Trainer, evaluate_predictions, predict, resolve_device  # noqa: E402
from src.explainability import (  # noqa: E402
    GradCAMPlusPlus,
    overlay_heatmap,
    plot_confusion_matrices,
    plot_learned_splines,
    plot_pareto_efficiency,
    plot_roc_pr_curves,
    plot_training_dynamics,
    plot_tsne,
    save_figure,
    set_ieee_style,
)
from src.losses import CompositeKANLoss, compute_class_weights  # noqa: E402
from src.models import (  # noqa: E402
    ABLATION_MODELS,
    BASELINE_MODELS,
    MODEL_SPECS,
    PROPOSED_MODEL,
    build_model,
    get_gradcam_target_layer,
    get_head_module_names,
    get_model_spec,
)
from src.utils import MetricTracker, get_logger, profile_model, set_seed  # noqa: E402

import export_latex_tables  # noqa: E402  (same directory)


# --------------------------------------------------------------------- config
def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        deep_update(cfg, overrides)
    return cfg


def experiment_dirs(cfg: Dict[str, Any], root: Path = ROOT) -> Dict[str, Path]:
    out = Path(cfg["experiment"].get("output_dir", "results"))
    out = out if out.is_absolute() else root / out
    ckpt = Path(cfg["experiment"].get("checkpoint_dir", "checkpoints"))
    ckpt = ckpt if ckpt.is_absolute() else root / ckpt
    dirs = {
        "out": out,
        "metrics": out / "metrics",
        "figures": out / "figures",
        "tables": out / "tables",
        "logs": out / "logs",
        "checkpoints": ckpt,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def all_model_names(cfg: Dict[str, Any]) -> List[str]:
    exp = cfg.get("experiments", {})
    names = [exp.get("proposed", PROPOSED_MODEL)]
    names += list(exp.get("baselines", BASELINE_MODELS))
    names += list(exp.get("ablations", ABLATION_MODELS))
    return names


def _load_model(name: str, cfg: Dict[str, Any], num_classes: int, dirs: Dict[str, Path], device: torch.device):
    model = build_model(name, num_classes, pretrained=False, image_size=int(cfg["data"]["image_size"]),
                        model_cfg=cfg.get("model", {}))
    ckpt = torch.load(dirs["checkpoints"] / f"{name}_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def _save_predictions(path: Path, preds: Dict[str, np.ndarray], class_names: List[str]) -> None:
    df = pd.DataFrame(preds["probs"], columns=[f"prob_{c}" for c in class_names])
    df.insert(0, "label", preds["labels"])
    df.insert(1, "pred", preds["probs"].argmax(1))
    df.to_csv(path, index=False, float_format="%.6f")


def _load_predictions(path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    return {"labels": df["label"].to_numpy(np.int64), "probs": df[prob_cols].to_numpy(np.float64)}


# --------------------------------------------------------------- one model
def run_single_experiment(
    name: str,
    cfg: Dict[str, Any],
    bundle,
    loaders,
    device: torch.device,
    dirs: Dict[str, Path],
    logger,
    skip_profiling: bool = False,
) -> Dict[str, Any]:
    """Train, select, evaluate (and optionally profile) one registry model."""
    spec = get_model_spec(name)
    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed, deterministic=bool(cfg["experiment"].get("deterministic", True)))
    image_size = int(cfg["data"]["image_size"])
    logger.info("=== %s (%s) ===", spec.display_name, name)

    model = build_model(name, bundle.num_classes, pretrained=bool(cfg["model"].get("pretrained", True)),
                        image_size=image_size, model_cfg=cfg.get("model", {}))
    loss_cfg = cfg.get("loss", {})
    weights = compute_class_weights(bundle.class_counts, loss_cfg.get("class_weight_method", "effective_number"),
                                    float(loss_cfg.get("class_weight_beta", 0.999)))
    use_reg = spec.use_spline_regularization
    loss_fn = CompositeKANLoss(
        class_weights=weights,
        gamma=float(loss_cfg.get("gamma", 2.0)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.05)),
        lambda_l1=float(loss_cfg.get("lambda_l1", 1.0)) if use_reg else 0.0,
        lambda_ent=float(loss_cfg.get("lambda_ent", 0.1)) if use_reg else 0.0,
    )
    train_cfg = copy.deepcopy(cfg["training"])
    train_cfg["seed"] = seed
    trainer = Trainer(model, loss_fn, train_cfg, device, checkpoint_path=dirs["checkpoints"] / f"{name}_best.pt",
                      logger=logger, model_name=name, num_classes=bundle.num_classes)
    history = trainer.fit(loaders["train"], loaders["val"])
    pd.DataFrame(history).to_csv(dirs["metrics"] / f"{name}_history.csv", index=False)

    eval_cfg = cfg.get("evaluation", {})
    val_pred = predict(trainer.model, loaders["val"], device)
    test_pred = predict(trainer.model, loaders["test"], device, return_embeddings=(name == PROPOSED_MODEL))
    _save_predictions(dirs["metrics"] / f"{name}_test_predictions.csv", test_pred, bundle.class_names)
    if "embeddings" in test_pred:
        np.save(dirs["checkpoints"] / f"{name}_test_embeddings.npy", test_pred["embeddings"])
    report = evaluate_predictions(
        test_pred["labels"], test_pred["probs"], bundle.class_names,
        n_bootstrap=int(eval_cfg.get("n_bootstrap", 1000)), ece_bins=int(eval_cfg.get("ece_bins", 15)), seed=seed,
        cal_labels=val_pred["labels"], cal_probs=val_pred["probs"],
        conformal_alpha=float(eval_cfg.get("conformal_alpha", 0.1)),
    )
    report.update(
        model=name,
        display_name=spec.display_name,
        family=spec.family,
        best_epoch=trainer.best_epoch,
        epochs_run=len(history),
        epochs_budget=int(cfg["training"]["epochs"]),
        train_time_min=trainer.total_time_s / 60.0,
        best_val_macro_f1=100 * max(h["val_macro_f1"] for h in history),
        dataset=bundle.source,
        image_size=image_size,
        spline_regularization=use_reg,
        lambda_l1=loss_fn.lambda_l1,
        lambda_ent=loss_fn.lambda_ent,
    )
    if hasattr(trainer.model, "kan_statistics"):
        report.update(trainer.model.kan_statistics())
    MetricTracker.save_json(report, dirs["metrics"] / f"{name}_metrics.json")
    logger.info("  test: acc=%.2f  bacc=%.2f  macro-F1=%.2f [%.2f, %.2f]  AUROC=%.2f  ECE=%.2f",
                report["accuracy"], report["balanced_accuracy"], report["macro_f1"],
                report["macro_f1_ci_lower"], report["macro_f1_ci_upper"], report["auroc"], report["ece"])
    if not skip_profiling:
        run_profiling(name, cfg, trainer.model, dirs, device, logger)
    return report


def run_profiling(name: str, cfg: Dict[str, Any], model: torch.nn.Module, dirs: Dict[str, Path],
                  device: torch.device, logger) -> Dict[str, Any]:
    eval_cfg = cfg.get("evaluation", {})
    hw = profile_model(
        model, get_head_module_names(model, name), image_size=int(cfg["data"]["image_size"]), device=device,
        warmup=int(eval_cfg.get("latency_warmup", 20)), runs=int(eval_cfg.get("latency_runs", 100)),
        throughput_batch_size=int(eval_cfg.get("throughput_batch_size", 32)),
    )
    ckpt = dirs["checkpoints"] / f"{name}_best.pt"
    if ckpt.exists():
        hw["checkpoint_file_mb"] = ckpt.stat().st_size / (1024**2)
    MetricTracker.save_json(hw, dirs["metrics"] / f"{name}_hardware.json")
    logger.info("  hardware: %.3fM params, head %.1fK, %.3f GFLOPs, %.2f ms/img, %.0f img/s",
                hw["params_m"], hw["head_params_k"], hw["gflops"], hw["latency_ms"], hw["throughput_img_s"])
    return hw


# --------------------------------------------------------------- aggregation
def build_summary(names: List[str], dirs: Dict[str, Path]) -> pd.DataFrame:
    rows, per_class_rows = [], []
    for name in names:
        mpath = dirs["metrics"] / f"{name}_metrics.json"
        hpath = dirs["metrics"] / f"{name}_hardware.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        h = json.loads(hpath.read_text()) if hpath.exists() else {}
        row = {k: v for k, v in m.items() if k not in ("per_class", "confusion_matrix")}
        row.update({k: v for k, v in h.items()})
        rows.append(row)
        for pc in m["per_class"]:
            per_class_rows.append({"model": name, "display_name": m["display_name"], **pc})
    summary = pd.DataFrame(rows)
    if len(summary):
        front = ["model", "display_name", "family"]
        summary = summary[front + [c for c in summary.columns if c not in front]]
        summary.to_csv(dirs["metrics"] / "all_experiments_summary.csv", index=False, float_format="%.6f")
        MetricTracker.save_json(summary.to_dict(orient="records"), dirs["metrics"] / "all_experiments_summary.json")
        pd.DataFrame(per_class_rows).to_csv(dirs["metrics"] / "all_per_class_metrics.csv", index=False, float_format="%.4f")
    return summary


def _select_xai_samples(labels: np.ndarray, probs: np.ndarray, class_names: List[str], n: int = 4) -> List[int]:
    """Highest-confidence correct predictions of diverse classes plus one error case."""
    preds = probs.argmax(1)
    conf = probs.max(1)
    priority = [c for c in ("mel", "bcc", "bkl", "nv", "akiec", "vasc", "df") if c in class_names]
    priority += [c for c in class_names if c not in priority]
    chosen: List[int] = []
    for cname in priority:
        if len(chosen) >= n - 1:
            break
        c = class_names.index(cname)
        idx = np.where((labels == c) & (preds == c))[0]
        if len(idx):
            chosen.append(int(idx[np.argmax(conf[idx])]))
    wrong = np.where(preds != labels)[0]
    wrong = [i for i in wrong if labels[i] != class_names.index("nv")] if "nv" in class_names else list(wrong)
    if len(wrong):
        chosen.append(int(wrong[np.argmax(conf[wrong])]))
    for i in range(len(labels)):
        if len(chosen) >= n:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:n]


def make_gradcam_figure(cfg, bundle, dirs, device, best_baseline: Optional[str], logger) -> None:
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    proposed = _load_model(PROPOSED_MODEL, cfg, bundle.num_classes, dirs, device)
    preds = _load_predictions(dirs["metrics"] / f"{PROPOSED_MODEL}_test_predictions.csv")
    idx = _select_xai_samples(preds["labels"], preds["probs"], bundle.class_names, int(cfg.get("xai", {}).get("num_samples", 4)))
    images = torch.stack([bundle.test[i][0] for i in idx]).to(device)
    labels = [int(bundle.test[i][1]) for i in idx]

    with GradCAMPlusPlus(proposed, get_gradcam_target_layer(proposed)) as cam_p:
        cams_p, logits_p = cam_p(images)
    with torch.no_grad():
        proposed(images)
        att = proposed.attention.last_attention_map.unsqueeze(1)
        att = F.interpolate(att, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        att = att.cpu().numpy()
    att = (att - att.min(axis=(1, 2), keepdims=True)) / (np.ptp(att, axis=(1, 2), keepdims=True) + 1e-8)
    probs_p = torch.softmax(logits_p, 1).cpu().numpy()

    cams_b, probs_b = None, None
    if best_baseline is not None and (dirs["checkpoints"] / f"{best_baseline}_best.pt").exists():
        base = _load_model(best_baseline, cfg, bundle.num_classes, dirs, device)
        with GradCAMPlusPlus(base, get_gradcam_target_layer(base)) as cam_b:
            cams_b, logits_b = cam_b(images)
        probs_b = torch.softmax(logits_b, 1).cpu().numpy()

    set_ieee_style()
    n_cols = 4 if cams_b is not None else 3
    fig, axes = plt.subplots(len(idx), n_cols, figsize=(7.16 * n_cols / 4, 1.85 * len(idx)), squeeze=False)
    names = bundle.class_names
    for r in range(len(idx)):
        img = denormalize(images[r].cpu()).permute(1, 2, 0).numpy()
        pred_p = int(probs_p[r].argmax())
        mark = "correct" if pred_p == labels[r] else "error"
        panels = [
            (img, f"GT: {names[labels[r]]} | Pred: {names[pred_p]} ({100 * probs_p[r, pred_p]:.0f}%, {mark})"),
            (overlay_heatmap(img, att[r], 0.5, "viridis"), "Att-ConvKAN coordinate-channel attention"),
            (overlay_heatmap(img, cams_p[r]), "Att-ConvKAN Grad-CAM++"),
        ]
        if cams_b is not None:
            pb = int(probs_b[r].argmax())
            panels.append((overlay_heatmap(img, cams_b[r]),
                           f"{MODEL_SPECS[best_baseline].display_name} Grad-CAM++ (pred: {names[pb]})"))
        for c, (im, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_title(title, fontsize=6.2)
            ax.axis("off")
    fig.tight_layout()
    save_figure(fig, dirs["figures"], "fig1_gradcam_comparison")
    logger.info("  fig1 written (test indices %s)", idx)


def make_spline_figure(cfg, bundle, loaders, dirs, device, logger) -> None:
    proposed = _load_model(PROPOSED_MODEL, cfg, bundle.num_classes, dirs, device)
    layer = proposed.head.kan1
    captured: List[torch.Tensor] = []
    handle = layer.register_forward_pre_hook(lambda m, inp: captured.append(inp[0].detach().cpu()))
    with torch.no_grad():
        for i, (images, _) in enumerate(loaders["test"]):
            proposed(images.to(device))
            if i >= 3:
                break
    handle.remove()
    inputs = torch.cat(captured)
    rows, cols = cfg.get("xai", {}).get("spline_grid", [3, 4])
    plot_learned_splines(layer, dirs["figures"], "fig2_learned_kan_splines", grid_shape=(rows, cols),
                         layer_name="KAN head layer 1 (576 -> 64)", inputs=inputs)
    plot_learned_splines(proposed.head.kan2, dirs["figures"], "fig2b_learned_kan_splines_output",
                         grid_shape=(rows, cols), layer_name="KAN head layer 2 (64 -> classes)",
                         out_names=bundle.class_names)
    logger.info("  fig2 written")


def aggregate(cfg: Dict[str, Any], bundle, loaders, device: torch.device, dirs: Dict[str, Path], logger,
              names: List[str], skip_xai: bool = False) -> pd.DataFrame:
    """Profile missing models, merge metrics, export tables and draw all figures."""
    for name in names:
        if (dirs["metrics"] / f"{name}_metrics.json").exists() and not (dirs["metrics"] / f"{name}_hardware.json").exists():
            if (dirs["checkpoints"] / f"{name}_best.pt").exists():
                logger.info("profiling %s", name)
                run_profiling(name, cfg, _load_model(name, cfg, bundle.num_classes, dirs, device), dirs, device, logger)
    summary = build_summary(names, dirs)
    if summary.empty:
        logger.warning("No finished experiments found; nothing to aggregate.")
        return summary
    export_latex_tables.export_all(dirs["metrics"], dirs["tables"])
    logger.info("LaTeX tables written to %s", dirs["tables"])

    idx = summary.set_index("model")
    baselines = [m for m in BASELINE_MODELS if m in idx.index]
    best_baseline = max(baselines, key=lambda m: idx.loc[m, "macro_f1"]) if baselines else None
    have_prop = PROPOSED_MODEL in idx.index

    if have_prop:
        p = _load_predictions(dirs["metrics"] / f"{PROPOSED_MODEL}_test_predictions.csv")
        plot_roc_pr_curves(p["labels"], p["probs"], bundle.class_names, dirs["figures"], "fig3_roc_and_pr_curves",
                           title_prefix="Att-ConvKAN: ")
        if best_baseline:
            b = _load_predictions(dirs["metrics"] / f"{best_baseline}_test_predictions.csv")
            plot_roc_pr_curves(b["labels"], b["probs"], bundle.class_names, dirs["figures"],
                               "fig3b_roc_and_pr_curves_best_baseline",
                               title_prefix=f"{MODEL_SPECS[best_baseline].display_name}: ")
        cms = {"Att-ConvKAN (ours)": np.array(json.loads((dirs["metrics"] / f"{PROPOSED_MODEL}_metrics.json").read_text())["confusion_matrix"])}
        if best_baseline:
            cms[f"{MODEL_SPECS[best_baseline].display_name} (best baseline)"] = np.array(
                json.loads((dirs["metrics"] / f"{best_baseline}_metrics.json").read_text())["confusion_matrix"])
        plot_confusion_matrices(cms, bundle.class_names, dirs["figures"], "fig4_confusion_matrix")

    if "gflops" in summary.columns:
        pareto_rows = summary[summary["family"].isin(["baseline", "proposed"]) | summary["model"].isin([ABLATION_MODELS[0]])]
        plot_pareto_efficiency(pareto_rows.to_dict(orient="records"), dirs["figures"], "fig5_pareto_efficiency",
                               highlight=PROPOSED_MODEL)

    histories = {}
    for name in baselines + ([PROPOSED_MODEL] if have_prop else []):
        hpath = dirs["metrics"] / f"{name}_history.csv"
        if hpath.exists():
            histories[MODEL_SPECS[name].display_name] = pd.read_csv(hpath).to_dict(orient="records")
    plot_training_dynamics(histories, dirs["figures"], "fig6_training_dynamics",
                           highlight=MODEL_SPECS[PROPOSED_MODEL].display_name)
    abl_hist = {}
    for name in ABLATION_MODELS + [PROPOSED_MODEL]:
        hpath = dirs["metrics"] / f"{name}_history.csv"
        if hpath.exists():
            abl_hist[MODEL_SPECS[name].display_name] = pd.read_csv(hpath).to_dict(orient="records")
    if abl_hist:
        plot_training_dynamics(abl_hist, dirs["figures"], "fig6b_training_dynamics_ablations",
                               highlight=MODEL_SPECS[PROPOSED_MODEL].display_name)

    if have_prop and not skip_xai and (dirs["checkpoints"] / f"{PROPOSED_MODEL}_best.pt").exists():
        make_gradcam_figure(cfg, bundle, dirs, device, best_baseline, logger)
        make_spline_figure(cfg, bundle, loaders, dirs, device, logger)
        emb_path = dirs["checkpoints"] / f"{PROPOSED_MODEL}_test_embeddings.npy"
        if emb_path.exists():
            p = _load_predictions(dirs["metrics"] / f"{PROPOSED_MODEL}_test_predictions.csv")
            plot_tsne(np.load(emb_path), p["labels"], bundle.class_names, dirs["figures"], "fig7_tsne_embeddings",
                      max_points=int(cfg.get("xai", {}).get("tsne_max_points", 2000)),
                      title="t-SNE of Att-ConvKAN test embeddings")
    logger.info("Figures written to %s", dirs["figures"])
    return summary


# ----------------------------------------------------------------------- main
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "quick_benchmark.yaml"))
    p.add_argument("--models", nargs="*", default=None, help="subset of registry names to run")
    p.add_argument("--force", action="store_true", help="re-run models that already have results")
    p.add_argument("--aggregate-only", action="store_true", help="skip training; merge results and draw figures")
    p.add_argument("--no-aggregate", action="store_true", help="train only, do not aggregate")
    p.add_argument("--skip-profiling", action="store_true", help="defer hardware profiling to aggregation")
    p.add_argument("--skip-xai", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--num-threads", type=int, default=None)
    p.add_argument("--dataset", default=None, help="dermamnist | pathmnist | pneumoniamnist | imagefolder | ham10000")
    p.add_argument("--data-dir", default=None, help="root of a local ImageFolder / raw HAM10000 directory")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--backbone", default=None, help="timm backbone of the Att-ConvKAN family")
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    args = parse_args(argv)
    overrides: Dict[str, Any] = {}
    if args.dataset:
        overrides.setdefault("data", {})["dataset"] = args.dataset
    if args.data_dir:
        overrides.setdefault("data", {})["data_dir"] = args.data_dir
    if args.image_size:
        overrides.setdefault("data", {}).update(image_size=args.image_size,
                                                native_size=args.image_size if args.image_size in (28, 64, 128, 224) else 224)
    if args.epochs:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.backbone:
        overrides.setdefault("model", {})["backbone"] = args.backbone
    if args.output_dir:
        overrides.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.device:
        overrides.setdefault("experiment", {})["device"] = args.device
    cfg = load_config(args.config, overrides)
    data_root = Path(cfg["data"].get("root", "data"))
    cfg["data"]["root"] = str(data_root if data_root.is_absolute() else ROOT / data_root)

    dirs = experiment_dirs(cfg)
    logger = get_logger("attconvkan", dirs["logs"] / f"{cfg['experiment']['name']}.log")
    device = resolve_device(cfg["experiment"].get("device", "auto"))
    n_threads = args.num_threads or cfg["experiment"].get("num_threads")
    if n_threads and device.type == "cpu":
        torch.set_num_threads(int(n_threads))
    logger.info("config=%s device=%s threads=%d", args.config, device, torch.get_num_threads())

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed, deterministic=bool(cfg["experiment"].get("deterministic", True)))
    bundle, loaders = build_dataloaders(cfg["data"], seed)
    logger.info("dataset=%s classes=%s train=%d val=%d test=%d counts=%s", bundle.source, bundle.class_names,
                len(bundle.train), len(bundle.val), len(bundle.test), bundle.class_counts)
    MetricTracker.save_json({"config": cfg, "class_names": bundle.class_names, "class_counts": bundle.class_counts,
                             "split_sizes": {k: len(getattr(bundle, k)) for k in ("train", "val", "test")},
                             "extra": bundle.extra},
                            dirs["metrics"] / f"run_config_{cfg['experiment']['name']}.json")

    names = args.models or all_model_names(cfg)
    unknown = [n for n in names if n not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")
    if not args.aggregate_only:
        for name in names:
            if (dirs["metrics"] / f"{name}_metrics.json").exists() and not args.force:
                logger.info("skipping %s (results exist; use --force to re-run)", name)
                continue
            t0 = time.time()
            run_single_experiment(name, cfg, bundle, loaders, device, dirs, logger, skip_profiling=args.skip_profiling)
            logger.info("  %s finished in %.1f min", name, (time.time() - t0) / 60)
    if args.no_aggregate:
        return pd.DataFrame()
    summary = aggregate(cfg, bundle, loaders, device, dirs, logger, all_model_names(cfg) if args.models is None or args.aggregate_only else names,
                        skip_xai=args.skip_xai)
    if not summary.empty:
        cols = ["display_name", "accuracy", "balanced_accuracy", "macro_f1", "auroc", "ece", "params_m", "head_params_k", "gflops", "latency_ms"]
        cols = [c for c in cols if c in summary.columns]
        logger.info("\n%s", summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return summary


if __name__ == "__main__":
    main()
