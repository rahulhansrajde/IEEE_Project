#!/usr/bin/env python
"""Convert experiment results into IEEEtran ``booktabs`` LaTeX tables.

Reads ``results/metrics/all_experiments_summary.csv`` and
``results/metrics/all_per_class_metrics.csv`` and writes

* ``results/tables/table1_baseline_comparison.tex`` (``table*``; best value per
  column in bold, second best underlined),
* ``results/tables/table2_ablation_study.tex`` (``table``),
* ``results/tables/table3_per_class_metrics.tex`` (``table``),
* ``results/tables/results_macros.tex`` -- ``\\res{<model>}{<metric>}`` macros
  so that the manuscript text always quotes the numbers that were measured.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.baselines import ABLATION_MODELS, BASELINE_MODELS, MODEL_SPECS, PROPOSED_MODEL  # noqa: E402

__all__ = [
    "rank_format",
    "make_table1",
    "make_table2",
    "make_table3",
    "make_macros",
    "export_all",
]

DATASET_LABELS = {
    "dermamnist": "DermaMNIST (HAM10000)",
    "pathmnist": "PathMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "imagefolder": "custom ImageFolder dataset",
    "ham10000": "HAM10000",
}


def _latex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(a, b)
    return text


def rank_format(values: Sequence[float], fmt: str, higher_is_better: bool = True) -> List[str]:
    """Format numbers, bolding the best and underlining the second-best value."""
    arr = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(arr)
    out = ["--" if not v else fmt.format(x) for x, v in zip(arr, valid)]
    if valid.sum() == 0:
        return out
    rounded = np.array([float(fmt.format(x)) if v else np.nan for x, v in zip(arr, valid)])
    key = rounded if higher_is_better else -rounded
    uniq = np.unique(key[valid])[::-1]
    best = uniq[0]
    second = uniq[1] if len(uniq) > 1 else None
    for i in range(len(arr)):
        if not valid[i]:
            continue
        if key[i] == best:
            out[i] = rf"\textbf{{{out[i]}}}"
        elif second is not None and key[i] == second:
            out[i] = rf"\underline{{{out[i]}}}"
    return out


def _ordered(df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    present = [n for n in names if n in set(df["model"])]
    return df.set_index("model").loc[present].reset_index()


def _setting_caption(df: pd.DataFrame) -> str:
    row = df.iloc[0]
    ds = DATASET_LABELS.get(str(row.get("dataset", "dermamnist")), str(row.get("dataset", "")))
    size = int(row.get("image_size", 0)) if not pd.isna(row.get("image_size", np.nan)) else 0
    n_test = int(row.get("n_test", 0)) if not pd.isna(row.get("n_test", np.nan)) else 0
    epochs = int(row.get("epochs_budget", 0)) if not pd.isna(row.get("epochs_budget", np.nan)) else 0
    device = str(row.get("device", "cpu")).upper()
    return (f"{ds} test set ($n={n_test}$), ${size}\\times{size}$ inputs, {epochs}-epoch budget. "
            f"Latency: batch size 1 on {device}.")


def make_table1(summary: pd.DataFrame) -> str:
    names = BASELINE_MODELS + [PROPOSED_MODEL]
    df = _ordered(summary, names)
    cols = [
        ("accuracy", "{:.2f}", True),
        ("balanced_accuracy", "{:.2f}", True),
        ("macro_f1", "{:.2f}", True),
        ("auroc", "{:.2f}", True),
        ("cohen_kappa", "{:.3f}", True),
        ("ece", "{:.2f}", False),
        ("ece_temperature_scaled", "{:.2f}", False),
        ("params_m", "{:.2f}", False),
        ("head_params_k", "{:.1f}", False),
        ("gflops", "{:.3f}", False),
        ("latency_ms", "{:.2f}", False),
    ]
    for c, _, _ in cols:
        if c not in df.columns:
            df[c] = np.nan
    formatted = {c: rank_format(df[c].to_numpy(dtype=float), f, hib) for c, f, hib in cols}
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Comparison with state-of-the-art baselines. " + _setting_caption(df)
        + r" Macro-F1 is reported with its 95\% bootstrap confidence interval. "
        r"Best value per column in \textbf{bold}, second best \underline{underlined}; "
        r"$\uparrow$/$\downarrow$: higher/lower is better.}",
        r"\label{tab:baselines}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l ccccc cc cccc}",
        r"\toprule",
        r"& \multicolumn{7}{c}{\textit{Diagnostic performance and calibration}} & \multicolumn{4}{c}{\textit{Efficiency}} \\",
        r"\cmidrule(lr){2-8} \cmidrule(lr){9-12}",
        r"Model & Acc.\,(\%)$\uparrow$ & B-Acc.\,(\%)$\uparrow$ & Macro-F1\,(\%)$\uparrow$ [95\% CI] & "
        r"AUROC\,(\%)$\uparrow$ & $\kappa\uparrow$ & ECE\,(\%)$\downarrow$ & ECE$_{TS}$\,(\%)$\downarrow$ & Params\,(M)$\downarrow$ & "
        r"Head\,(K)$\downarrow$ & GFLOPs$\downarrow$ & Lat.\,(ms)$\downarrow$ \\",
        r"\midrule",
    ]
    for i, row in df.iterrows():
        name = _latex_escape(MODEL_SPECS[row["model"]].display_name)
        if row["model"] == PROPOSED_MODEL:
            lines.append(r"\midrule")
        ci = f"[{row['macro_f1_ci_lower']:.1f}, {row['macro_f1_ci_upper']:.1f}]"
        cells = [
            formatted["accuracy"][i],
            formatted["balanced_accuracy"][i],
            f"{formatted['macro_f1'][i]} {{\\scriptsize {ci}}}",
            formatted["auroc"][i],
            formatted["cohen_kappa"][i],
            formatted["ece"][i],
            formatted["ece_temperature_scaled"][i],
            formatted["params_m"][i],
            formatted["head_params_k"][i],
            formatted["gflops"][i],
            formatted["latency_ms"][i],
        ]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    return "\n".join(lines)


def make_table2(summary: pd.DataFrame) -> str:
    names = ABLATION_MODELS + [PROPOSED_MODEL]
    df = _ordered(summary, names)
    ref = df.loc[df["model"] == ABLATION_MODELS[0], "macro_f1"]
    ref_f1 = float(ref.iloc[0]) if len(ref) else np.nan
    cols = [
        ("balanced_accuracy", "{:.2f}", True),
        ("macro_f1", "{:.2f}", True),
        ("auroc", "{:.2f}", True),
        ("ece", "{:.2f}", False),
        ("head_params_k", "{:.1f}", False),
    ]
    formatted = {c: rank_format(df[c].to_numpy(dtype=float), f, hib) for c, f, hib in cols}
    check, cross = r"\checkmark", r"--"
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Ablation study of the Att-ConvKAN components (same backbone, data, budget and seed). "
        r"$\Delta$F1: macro-F1 change w.r.t.\ the backbone + MLP-head reference (row 1). "
        r"Symb.: share of KAN-head edges whose spline term is $<1\%$ of their SiLU weight, i.e.\ edges that "
        r"reduce to the symbolic form $w_b\,\mathrm{SiLU}(x)$; $H$: normalised entropy of the spline edge "
        r"importances (lower = more concentrated).}",
        r"\label{tab:ablation}",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{cccc cccc c cc c}",
        r"\toprule",
        r"\rotatebox{90}{KANConv} & \rotatebox{90}{Attention} & \rotatebox{90}{KAN head} & "
        r"\rotatebox{90}{Spline reg.} & B-Acc. & Macro-F1 & AUROC & ECE$\downarrow$ & "
        r"Head (K) & Symb. (\%) & $H$ & $\Delta$F1 \\",
        r"\midrule",
    ]
    for i, row in df.iterrows():
        comps = MODEL_SPECS[row["model"]].components
        flags = [check if comps.get(k) else cross for k in ("kan_conv", "attention", "kan_head", "spline_reg")]
        delta = row["macro_f1"] - ref_f1
        delta_s = "--" if row["model"] == ABLATION_MODELS[0] or np.isnan(delta) else f"{delta:+.2f}"
        sym = row.get("kan_edge_sparsity", np.nan)
        ent = row.get("kan_spline_entropy", np.nan)
        spline_cells = ["--", "--"] if pd.isna(sym) else [f"{100 * sym:.1f}", f"{ent:.3f}"]
        if row["model"] == PROPOSED_MODEL:
            lines.append(r"\midrule")
        cells = flags + [formatted[c][i] for c, _, _ in cols] + spline_cells + [delta_s]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def make_table3(per_class: pd.DataFrame, model: str = PROPOSED_MODEL, descriptions: Optional[Dict[str, str]] = None) -> str:
    df = per_class[per_class["model"] == model].reset_index(drop=True)
    cols = [("precision", True), ("recall", True), ("f1", True), ("specificity", True), ("auroc", True)]
    formatted = {c: rank_format(df[c].to_numpy(dtype=float), "{:.2f}", hib) for c, hib in cols}
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Per-class test performance (\%) of " + _latex_escape(MODEL_SPECS[model].display_name)
        + r". Best class per column in \textbf{bold}, second \underline{underlined}.}",
        r"\label{tab:perclass}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l r ccccc}",
        r"\toprule",
        r"Class & $n$ & Prec. & Rec./Sens. & F1 & Spec. & AUROC \\",
        r"\midrule",
    ]
    for i, row in df.iterrows():
        name = _latex_escape(str(row["class"]))
        if descriptions and row["class"] in descriptions:
            name = rf"{name}\textsuperscript{{\,}}"
        cells = [str(int(row["support"]))] + [formatted[c][i] for c, _ in cols]
        lines.append(f"\\texttt{{{name}}} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    macro = [f"{df[c].mean():.2f}" for c, _ in cols]
    lines.append(r"Macro avg. & " + str(int(df["support"].sum())) + " & " + " & ".join(macro) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def make_macros(summary: pd.DataFrame, per_class: Optional[pd.DataFrame] = None,
                paired: Optional[pd.DataFrame] = None) -> str:
    """``\\res{model}{metric}`` look-up macros for every numeric summary value."""
    lines = [
        "% Auto-generated by scripts/export_latex_tables.py -- do not edit by hand.",
        r"\makeatletter",
        r"\providecommand{\res}[2]{\@ifundefined{res@#1@#2}{\textbf{??}}{\csname res@#1@#2\endcsname}}",
    ]
    fmt = {
        "cohen_kappa": "{:.3f}", "params_m": "{:.2f}", "gflops": "{:.3f}", "head_params_k": "{:.1f}",
        "model_size_mb": "{:.2f}", "latency_ms": "{:.2f}", "throughput_img_s": "{:.0f}",
        "conformal_set_size": "{:.2f}", "kan_edge_sparsity": "{:.3f}", "kan_spline_entropy": "{:.4f}",
        "kan_top_mass": "{:.3f}", "temperature": "{:.3f}", "kan_head_mean_l1": "{:.2e}", "kan_conv_mean_l1": "{:.2e}",
        "train_time_min": "{:.1f}", "best_epoch": "{:.0f}",
    }
    for _, row in summary.iterrows():
        for col, val in row.items():
            if col == "model" or not isinstance(val, (int, float, np.floating, np.integer)) or pd.isna(val):
                continue
            s = fmt.get(col, "{:.2f}").format(float(val))
            lines.append(rf"\expandafter\def\csname res@{row['model']}@{col}\endcsname{{{s}}}")
    for _, row in summary.iterrows():
        if "kan_edge_sparsity" in row and not pd.isna(row["kan_edge_sparsity"]):
            lines.append(rf"\expandafter\def\csname res@{row['model']}@kan_edge_sparsity_pct\endcsname"
                         rf"{{{100 * row['kan_edge_sparsity']:.2f}}}")
            lines.append(rf"\expandafter\def\csname res@{row['model']}@kan_head_edges_k\endcsname"
                         rf"{{{row['kan_head_edges'] / 1000:.1f}}}")
    # Derived quantities quoted in the manuscript.
    idx = summary.set_index("model")
    if PROPOSED_MODEL in idx.index:
        prop = idx.loc[PROPOSED_MODEL]
        derived = {}
        if ABLATION_MODELS[0] in idx.index:
            ref = idx.loc[ABLATION_MODELS[0]]
            derived["head_reduction_pct"] = 100 * (1 - prop["head_params_k"] / ref["head_params_k"])
            derived["f1_gain_over_mlp"] = prop["macro_f1"] - ref["macro_f1"]
        if ABLATION_MODELS[3] in idx.index:
            derived["reg_cost_f1"] = idx.loc[ABLATION_MODELS[3], "macro_f1"] - prop["macro_f1"]
        if paired is not None and len(paired):
            abl = paired[paired["model"].isin(ABLATION_MODELS)]
            if len(abl):
                derived["min_ablation_p"] = float(abl["p_value"].min())
        base = [m for m in BASELINE_MODELS if m in idx.index]
        if base:
            best = max(base, key=lambda m: idx.loc[m, "macro_f1"])
            derived["best_baseline_f1"] = idx.loc[best, "macro_f1"]
            derived["f1_gap_to_best_baseline"] = prop["macro_f1"] - idx.loc[best, "macro_f1"]
            derived["param_ratio_best_baseline"] = idx.loc[best, "params_m"] / prop["params_m"]
            derived["gflops_ratio_best_baseline"] = idx.loc[best, "gflops"] / prop["gflops"]
            lines.append(rf"\def\bestbaselinename{{{_latex_escape(MODEL_SPECS[best].display_name)}}}")
            if paired is not None and best in set(paired["model"]):
                p_best = float(paired.loc[paired["model"] == best, "p_value"].iloc[0])
                p_str = "<0.001" if p_best < 0.001 else f"={p_best:.3f}"
                lines.append(rf"\def\bestbaselinep{{{p_str}}}")
            if per_class is not None and len(per_class):
                mel = per_class[(per_class["model"] == best) & (per_class["class"] == "mel")]
                if len(mel):
                    derived["best_baseline_mel_recall"] = float(mel["recall"].iloc[0])
            lighter = [m for m in base if m != "mobilenetv3_small"]
            if lighter:
                derived["min_heavy_params_m"] = float(idx.loc[lighter, "params_m"].min())
                derived["max_heavy_params_m"] = float(idx.loc[lighter, "params_m"].max())
            heavy = idx.loc[base, "params_m"].idxmax()
            derived["param_ratio_largest_baseline"] = idx.loc[heavy, "params_m"] / prop["params_m"]
            lines.append(rf"\def\largestbaselinename{{{_latex_escape(MODEL_SPECS[heavy].display_name)}}}")
        for k, v in derived.items():
            lines.append(rf"\expandafter\def\csname res@derived@{k}\endcsname{{{v:.2f}}}")
    if paired is not None and len(paired):
        for _, row in paired.iterrows():
            for col in ("delta_macro_f1", "delta_macro_f1_ci_lower", "delta_macro_f1_ci_upper",
                        "delta_balanced_accuracy", "delta_balanced_accuracy_ci_lower",
                        "delta_balanced_accuracy_ci_upper", "p_value"):
                if col == "p_value":
                    val = "<0.001" if row[col] < 0.001 else f"={row[col]:.3f}"
                else:
                    val = f"{row[col]:+.2f}"
                lines.append(rf"\expandafter\def\csname res@paired@{row['model']}@{col}\endcsname{{{val}}}")
    if per_class is not None and len(per_class):
        for _, row in per_class.iterrows():
            for col in ("precision", "recall", "f1", "specificity", "auroc"):
                lines.append(rf"\expandafter\def\csname res@{row['model']}@{row['class']}_{col}\endcsname{{{row[col]:.2f}}}")
    lines.append(r"\makeatother")
    return "\n".join(lines) + "\n"


def export_all(metrics_dir: str | Path = ROOT / "results" / "metrics", tables_dir: str | Path = ROOT / "results" / "tables") -> Dict[str, Path]:
    metrics_dir, tables_dir = Path(metrics_dir), Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(metrics_dir / "all_experiments_summary.csv")
    per_class_path = metrics_dir / "all_per_class_metrics.csv"
    per_class = pd.read_csv(per_class_path) if per_class_path.exists() else None
    outputs = {
        "table1": tables_dir / "table1_baseline_comparison.tex",
        "table2": tables_dir / "table2_ablation_study.tex",
        "table3": tables_dir / "table3_per_class_metrics.tex",
        "macros": tables_dir / "results_macros.tex",
    }
    outputs["table1"].write_text(make_table1(summary))
    outputs["table2"].write_text(make_table2(summary))
    if per_class is not None:
        model = PROPOSED_MODEL if PROPOSED_MODEL in set(per_class["model"]) else per_class["model"].iloc[0]
        outputs["table3"].write_text(make_table3(per_class, model))
    paired_path = metrics_dir / "paired_bootstrap_vs_proposed.csv"
    paired = pd.read_csv(paired_path) if paired_path.exists() else None
    outputs["macros"].write_text(make_macros(summary, per_class, paired))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", default=str(ROOT / "results" / "metrics"))
    parser.add_argument("--tables-dir", default=str(ROOT / "results" / "tables"))
    args = parser.parse_args()
    for key, path in export_all(args.metrics_dir, args.tables_dir).items():
        print(f"{key:>7s}: {path}")


if __name__ == "__main__":
    main()
