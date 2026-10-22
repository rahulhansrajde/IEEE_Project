"""Integration tests: data loading, one training step, evaluation, profiling, LaTeX export,
and a complete miniature run of the experiment orchestrator."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

import export_latex_tables
import run_all_experiments
from src.data import MixupCutmix, build_dataloaders, build_transforms, stratified_split
from src.engine import (
    Trainer,
    conformal_calibrate,
    conformal_evaluate,
    evaluate_predictions,
    expected_calibration_error,
    predict,
)
from src.explainability import GradCAMPlusPlus
from src.losses import CompositeKANLoss, compute_class_weights
from src.models import build_model, get_gradcam_target_layer, get_head_module_names
from src.utils import profile_model, set_seed

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["akiec", "bcc", "mel", "nv"]


def _make_imagefolder(root: Path, per_class: int = 10, size: int = 40) -> Path:
    rng = np.random.default_rng(0)
    for ci, cname in enumerate(CLASSES):
        d = root / cname
        d.mkdir(parents=True, exist_ok=True)
        for k in range(per_class):
            base = np.zeros((size, size, 3), dtype=np.float64)
            base[..., ci % 3] = 0.6  # class-dependent colour cue
            img = np.clip(base + rng.normal(0, 0.15, base.shape), 0, 1)
            Image.fromarray((img * 255).astype(np.uint8)).save(d / f"{cname}_{k}.png")
    return root


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    return _make_imagefolder(tmp_path_factory.mktemp("imgs"))


def _data_cfg(image_dir: Path, image_size: int = 32):
    return {
        "dataset": "imagefolder",
        "data_dir": str(image_dir),
        "image_size": image_size,
        "batch_size": 8,
        "eval_batch_size": 16,
        "num_workers": 0,
        "split_fractions": [0.6, 0.2, 0.2],
    }


def test_imagefolder_loading_and_split(image_dir):
    bundle, loaders = build_dataloaders(_data_cfg(image_dir), seed=1)
    assert bundle.class_names == sorted(CLASSES)
    assert bundle.num_classes == 4
    assert len(bundle.train) + len(bundle.val) + len(bundle.test) == 40
    assert sum(bundle.class_counts) == len(bundle.train)
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (8, 3, 32, 32)
    assert labels.dtype == torch.int64


def test_stratified_split_is_disjoint_and_group_aware():
    labels = np.repeat(np.arange(3), 40)
    parts = stratified_split(labels, [0.7, 0.1, 0.2], seed=0)
    allidx = np.concatenate(parts)
    assert len(np.unique(allidx)) == len(labels)
    groups = np.arange(len(labels)) // 2
    gparts = stratified_split(labels, [0.7, 0.1, 0.2], seed=0, groups=groups)
    sets = [set(groups[p]) for p in gparts]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_transforms_and_mixup():
    tf = build_transforms(48, train=True, clahe=True)
    x = tf(Image.fromarray(np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)))
    assert x.shape == (3, 48, 48)
    mixer = MixupCutmix(prob=1.0, seed=0)
    imgs, ya, yb, lam = mixer(torch.randn(4, 3, 16, 16), torch.arange(4))
    assert imgs.shape == (4, 3, 16, 16) and 0.0 <= lam <= 1.0 and yb.shape == ya.shape


@pytest.mark.skipif(not (ROOT / "data" / "dermamnist_64.npz").exists(), reason="DermaMNIST-64 not cached")
def test_dermamnist_loading():
    cfg = {"dataset": "dermamnist", "root": str(ROOT / "data"), "image_size": 64, "native_size": 64,
           "num_workers": 0, "download": False, "max_samples_per_split": 50}
    bundle, loaders = build_dataloaders(cfg)
    assert bundle.num_classes == 7
    assert bundle.class_names[5] == "nv"
    x, y = next(iter(loaders["test"]))
    assert x.shape[1:] == (3, 64, 64)


def test_one_step_training_evaluation_and_export(image_dir, tmp_path):
    set_seed(0)
    bundle, loaders = build_dataloaders(_data_cfg(image_dir), seed=0)
    model = build_model("att_convkan_proposed", bundle.num_classes, pretrained=False, image_size=32)
    loss = CompositeKANLoss(compute_class_weights(bundle.class_counts), lambda_l1=1.0, lambda_ent=0.1)
    train_cfg = {"epochs": 2, "lr": 1e-3, "warmup_epochs": 1, "max_train_batches": 1, "grid_update_epochs": [1],
                 "grid_update_samples": 16, "early_stopping_patience": 5, "head_lr_multiplier": 2.0}
    ckpt = tmp_path / "ckpt.pt"
    trainer = Trainer(model, loss, train_cfg, torch.device("cpu"), checkpoint_path=ckpt, num_classes=4,
                      model_name="att_convkan_proposed")
    before = [p.detach().clone() for p in model.head.parameters()]
    history = trainer.fit(loaders["train"], loaders["val"])
    assert len(history) == 2 and ckpt.exists()
    assert any(not torch.equal(b, a) for b, a in zip(before, model.head.parameters()))
    for key in ("train_loss", "val_loss", "val_macro_f1", "val_auroc", "lr"):
        assert key in history[0]

    test = predict(model, loaders["test"], torch.device("cpu"), return_embeddings=True)
    val = predict(model, loaders["val"], torch.device("cpu"))
    assert np.allclose(test["probs"].sum(1), 1.0, atol=1e-5)
    report = evaluate_predictions(test["labels"], test["probs"], bundle.class_names, n_bootstrap=50,
                                  cal_labels=val["labels"], cal_probs=val["probs"])
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
                "macro_specificity", "auroc", "cohen_kappa", "ece", "macro_f1_ci_lower", "macro_f1_ci_upper",
                "conformal_coverage", "conformal_set_size"):
        assert key in report
    assert report["macro_f1_ci_lower"] <= report["macro_f1_ci_upper"]
    assert len(report["per_class"]) == 4

    hw = profile_model(model, get_head_module_names(model), image_size=32, warmup=2, runs=3, throughput_batch_size=4)
    assert hw["params_m"] > 0 and hw["head_params_k"] > 0 and hw["gflops"] > 0 and hw["latency_ms"] > 0

    with GradCAMPlusPlus(model, get_gradcam_target_layer(model)) as cam:
        maps, logits = cam(next(iter(loaders["test"]))[0][:2])
    assert maps.shape == (2, 32, 32) and maps.min() >= 0 and maps.max() <= 1 + 1e-6

    metrics_dir, tables_dir = tmp_path / "metrics", tmp_path / "tables"
    metrics_dir.mkdir()
    row = {k: v for k, v in report.items() if k not in ("per_class", "confusion_matrix")}
    rows = []
    for name, family in (("att_convkan_proposed", "proposed"), ("resnet50", "baseline"),
                         ("ablation_1_backbone_mlp", "ablation")):
        r = dict(row, **hw, model=name, family=family, display_name=name, dataset="imagefolder",
                 image_size=32, epochs_budget=2)
        r["macro_f1"] = r["macro_f1"] + len(rows)
        rows.append(r)
    pd.DataFrame(rows).to_csv(metrics_dir / "all_experiments_summary.csv", index=False)
    pd.DataFrame([dict(pc, model="att_convkan_proposed") for pc in report["per_class"]]).to_csv(
        metrics_dir / "all_per_class_metrics.csv", index=False)
    outputs = export_latex_tables.export_all(metrics_dir, tables_dir)
    t1 = outputs["table1"].read_text()
    assert r"\begin{table*}" in t1 and r"\toprule" in t1 and r"\bottomrule" in t1 and r"\textbf{" in t1
    assert r"\underline{" in t1
    assert r"\begin{table}" in outputs["table2"].read_text()
    assert "akiec" in outputs["table3"].read_text()
    assert r"\res" in outputs["macros"].read_text()


def test_metric_helpers():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    labels = np.array([0, 1, 1, 1])
    ece = expected_calibration_error(probs, labels, n_bins=15)
    assert 0.0 <= ece <= 1.0
    qhat = conformal_calibrate(probs, labels, alpha=0.25)
    res = conformal_evaluate(probs, labels, qhat)
    assert res["coverage"] >= 0.75 and res["avg_set_size"] >= 1.0


def test_rank_format_bolds_best_and_underlines_second():
    out = export_latex_tables.rank_format([1.0, 3.0, 2.0], "{:.1f}", higher_is_better=True)
    assert out == ["1.0", r"\textbf{3.0}", r"\underline{2.0}"]
    out = export_latex_tables.rank_format([1.0, 3.0, 2.0], "{:.1f}", higher_is_better=False)
    assert out == [r"\textbf{1.0}", "3.0", r"\underline{2.0}"]


def test_orchestrator_end_to_end(image_dir, tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "quick_benchmark.yaml").read_text())
    cfg["experiment"].update(output_dir=str(tmp_path / "results"), checkpoint_dir=str(tmp_path / "ckpt"),
                             name="pytest_run", num_threads=2)
    cfg["data"].update(_data_cfg(image_dir))
    cfg["model"]["pretrained"] = False
    cfg["training"].update(epochs=1, max_train_batches=1, grid_update_epochs=[])
    cfg["evaluation"].update(n_bootstrap=20, latency_warmup=1, latency_runs=2, throughput_batch_size=2)
    cfg["experiments"] = {"proposed": "att_convkan_proposed", "baselines": ["mobilenetv3_small"],
                          "ablations": ["ablation_1_backbone_mlp"]}
    cfg["xai"] = {"num_samples": 4, "tsne_max_points": 200, "spline_grid": [3, 4]}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    summary = run_all_experiments.main(["--config", str(cfg_path)])
    assert set(summary["model"]) == {"att_convkan_proposed", "mobilenetv3_small", "ablation_1_backbone_mlp"}
    res = tmp_path / "results"
    for rel in ("metrics/all_experiments_summary.csv", "metrics/all_experiments_summary.json",
                "tables/table1_baseline_comparison.tex", "tables/table2_ablation_study.tex",
                "tables/table3_per_class_metrics.tex"):
        assert (res / rel).stat().st_size > 0, rel
    for stem in ("fig1_gradcam_comparison", "fig2_learned_kan_splines", "fig3_roc_and_pr_curves",
                 "fig4_confusion_matrix", "fig5_pareto_efficiency", "fig6_training_dynamics",
                 "fig7_tsne_embeddings"):
        for ext in ("png", "pdf"):
            assert (res / "figures" / f"{stem}.{ext}").stat().st_size > 0, stem
