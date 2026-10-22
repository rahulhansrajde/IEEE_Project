# Att-ConvKAN: Attention-Guided Convolutional Kolmogorov-Arnold Networks for Parameter-Efficient and Explainable Medical Image Diagnosis

Research code, experiment results and IEEE manuscript for **Att-ConvKAN**, a hybrid CNN that combines

* a pretrained **MobileNetV3-Small** backbone (or any `timm` backbone, e.g. `convnext_tiny`),
* a **KANConv bottleneck**: depthwise 3x3 convolution whose kernel taps are learnable cubic B-spline functions,
* **coordinate-channel attention** (CBAM channel branch + coordinate spatial branch),
* a **KAN classification head** `GAP -> LN -> Dropout -> KANLinear(576, 64) -> LN -> KANLinear(64, K)`,
* a **composite loss**: class-balanced focal cross-entropy + spline L1 + spline-entropy regularisation.

All KAN layers are implemented in **pure, vectorised PyTorch** (exact Cox-de Boor recursion, least-squares
grid refinement), so they run on CPU, Apple-Silicon MPS and CUDA with full autograd and no custom kernels.

## Results (DermaMNIST / HAM10000, measured in this repository)

All numbers below were produced by `python scripts/run_all_experiments.py --config configs/quick_benchmark.yaml`
on a 4-core CPU (no GPU): official DermaMNIST split (7,007 / 1,003 / 2,005 images, 7 classes), 128x128 inputs,
**identical 10-epoch recipe and seed for every model**, best-validation-macro-F1 checkpoint evaluated once on the test set.
Latency: batch size 1 on the same CPU (4 threads). "paired p" is a paired bootstrap test of the macro-F1 difference
against Att-ConvKAN on the same test images. ECE(TS): after temperature scaling fitted on the validation set.

**Table I – baselines**

| Model | Acc. | B-Acc. | Macro-F1 [95% CI] | AUROC | ECE → ECE(TS) | Params (M) | Head (K) | GFLOPs | Latency (ms) | paired p vs. ours |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 82.24 | 75.00 | 71.96 [68.4, 75.2] | 95.63 | 19.0 → 2.4 | 23.52 | 14.3 | 2.669 | 48.0 | <0.001 |
| DenseNet-121 | 84.54 | 76.31 | 75.16 [71.4, 78.5] | 96.80 | 17.7 → 1.9 | 6.96 | 7.2 | 1.850 | 63.0 | <0.001 |
| MobileNetV3-S | 77.46 | 67.70 | 62.27 [58.5, 65.6] | 94.05 | 18.2 → 2.7 | 1.53 | 598.0 | 0.038 | 9.0 | 0.633 |
| ViT-Tiny | 83.09 | 79.27 | 74.43 [71.1, 77.5] | 96.55 | 21.3 → 1.6 | 5.50 | 1.4 | 0.709 | 14.8 | <0.001 |
| ConvNeXt-T | 82.49 | 76.16 | 72.35 [68.6, 75.6] | 96.41 | 20.3 → 2.8 | 27.83 | 6.9 | 2.909 | 52.1 | <0.001 |
| Att-ConvKAN (Ours) | 79.00 | 67.90 | 63.05 [59.5, 66.8] | 94.43 | 20.6 → 2.3 | 1.59 | 374.4 | 0.048 | 16.1 | — |

**Table II – ablations** (same MobileNetV3-S backbone)

| Model | Acc. | B-Acc. | Macro-F1 [95% CI] | AUROC | ECE → ECE(TS) | Params (M) | Head (K) | GFLOPs | Latency (ms) | paired p vs. ours |
|---|---|---|---|---|---|---|---|---|---|---|
| Backbone + MLP head | 79.30 | 69.07 | 64.67 [60.7, 68.3] | 94.65 | 20.5 → 2.5 | 1.53 | 599.2 | 0.038 | 10.5 | 0.374 |
| Backbone + KAN head | 79.15 | 70.12 | 65.57 [61.6, 69.1] | 95.19 | 20.0 → 3.8 | 1.30 | 374.4 | 0.037 | 11.2 | 0.162 |
| Backbone + Att. + MLP head | 78.15 | 70.90 | 62.87 [59.5, 66.1] | 94.73 | 17.7 → 2.7 | 1.63 | 599.2 | 0.038 | 10.0 | 0.938 |
| Full w/o spline reg. | 78.30 | 71.65 | 66.23 [62.4, 69.5] | 94.97 | 19.1 → 2.2 | 1.59 | 374.4 | 0.048 | 21.5 | 0.076 |
| Att-ConvKAN (Ours) | 79.00 | 67.90 | 63.05 [59.5, 66.8] | 94.43 | 20.6 → 2.3 | 1.59 | 374.4 | 0.048 | 16.1 | — |

### Honest summary of the findings

* **Att-ConvKAN is on par with its own MobileNetV3-S baseline** (63.05 vs. 62.27 macro-F1,
  p = 0.63) while its KAN head uses 37.5% fewer parameters than the 1024-unit MLP head
  (374.4 K vs. 598.0 K). It does **not** sit on the accuracy–size Pareto frontier:
  the backbone + MLP-head ablation is marginally smaller and slightly (not significantly) better.
* **Larger baselines are significantly more accurate** (all p < 0.001); DenseNet-121 is best with 75.16 macro-F1
  at 4.4x the parameters and ~39x the FLOPs.
* **Spline regularisation is a trade-off, not a free gain:** with λ1 = 1, λH = 0.1 it turns 99.99% of the KAN-head edges
  into closed-form `w·SiLU(x)` terms (normalised edge entropy 0.0006), but costs 3.18 macro-F1 points
  versus the unregularised full model (p = 0.076), which is the best lightweight variant.
  The weights were not tuned; tuning them on the validation set is the obvious next step.
* **FLOPs understate KAN cost on CPU:** the B-spline recursion raises single-image latency from 9.0 ms
  (MobileNetV3-S) to 16.1 ms, slower than ViT-Tiny (14.8 ms) despite ~15x fewer FLOPs.
* All models are strongly **under-confident** under focal loss + class weighting (ECE 17.7–21.3%); temperature scaling brings ECE to 1.6–3.8%.
* Most models were still improving at epoch 10, so these are short-budget results; `configs/default_config.yaml` holds the full
  50-epoch, 224x224 protocol intended for a GPU.

CPU note: subnormal-float flushing (`torch.set_flush_denormal(True)`) was enabled for the DenseNet-121 training run and for all
latency measurements after we found that subnormals slowed pretrained DenseNet-121 by ~6x on this CPU; the other nine models were
trained before the setting was added. It is now on by default for CPU runs (`experiment.flush_denormal`).

Figures (300-DPI PNG + PDF) are in `results/figures/`: Grad-CAM++ comparison (fig1), learned KAN splines (fig2, fig2b, fig2c),
ROC/PR curves (fig3), confusion matrices (fig4), Pareto efficiency (fig5), training dynamics (fig6), t-SNE (fig7).
LaTeX tables are in `results/tables/`, and the compiled manuscript is `paper/main.pdf`.

## Repository layout

```
configs/                  quick_benchmark.yaml (the run reported here), default_config.yaml (50 epochs, 224 px, GPU)
src/models/kan_layers.py  KANLinear, KANConv2d, KANConvBottleneck (B-splines, grid update, L1/entropy regularisers)
src/models/attention.py   ChannelAttention, CoordinateSpatialAttention, CoordinateChannelAttention
src/models/conv_kan_net.py  AttConvKANNet (+ ablation switches), KAN / MLP heads
src/models/baselines.py   build_model() registry: 5 baselines, 4 ablations, proposed model
src/losses/               CompositeKANLoss, FocalCrossEntropy, class-balanced weights
src/data/                 DermaMNIST/PathMNIST/PneumoniaMNIST auto-download, ImageFolder, raw HAM10000; CLAHE, Mixup/CutMix
src/engine/               Trainer (AMP, warm-up + cosine warm restarts, early stopping, grid updates), evaluator
src/explainability/       Grad-CAM++, learned-spline plots, IEEE-style ROC/PR, confusion, t-SNE, Pareto, training curves
src/utils/                seeding, rich logging + JSON metric tracking, FLOPs/latency profiler
scripts/run_all_experiments.py   master orchestrator (train -> evaluate -> profile -> figures -> LaTeX)
scripts/export_latex_tables.py   summary CSV -> booktabs tables + \res{model}{metric} macros
scripts/inference_demo.py        single-image diagnosis with Grad-CAM++ and spline explanation
tests/                    unit + integration tests (pytest)
results/metrics|figures|tables   generated results (tracked in git)
paper/main.tex            IEEEtran conference manuscript (inputs the generated tables/figures/macros)
```

## Installation

```bash
pip install -r requirements.txt          # CPU/MPS; for CUDA install the matching torch wheel first
pytest tests/ -v
```

## Reproducing the experiments

```bash
# All 10 models (proposed + 5 baselines + 4 ablations), figures and tables. DermaMNIST downloads automatically.
python scripts/run_all_experiments.py --config configs/quick_benchmark.yaml

# Full-resolution, 50-epoch protocol (GPU recommended)
python scripts/run_all_experiments.py --config configs/default_config.yaml

# Split the work over several processes, then merge (profiling is done quietly at merge time)
python scripts/run_all_experiments.py --models resnet50 densenet121 --skip-profiling --no-aggregate
python scripts/run_all_experiments.py --models att_convkan_proposed vit_tiny ... --skip-profiling --no-aggregate
python scripts/run_all_experiments.py --aggregate-only

# Other data sources
python scripts/run_all_experiments.py --dataset pathmnist
python scripts/run_all_experiments.py --dataset imagefolder --data-dir /path/to/folder   # class sub-folders or train/val/test
python scripts/run_all_experiments.py --dataset ham10000 --data-dir /path/to/HAM10000    # lesion-level split

# Regenerate LaTeX tables only
python scripts/export_latex_tables.py

# Explain a single image with a trained checkpoint
python scripts/inference_demo.py --sample-index 0
python scripts/inference_demo.py --image my_lesion.jpg
```

Finished models are skipped on re-runs (use `--force` to retrain). Checkpoints are written to `checkpoints/`
(git-ignored); metrics, predictions, histories, figures (300-DPI PNG + PDF) and tables go to `results/`.

## Building the paper

```bash
cd paper && pdflatex main.tex && pdflatex main.tex
```

`paper/main.tex` reads `../results/tables/*.tex` and `../results/figures/*.pdf`, and every number in the text
comes from `results/tables/results_macros.tex`, so re-running the pipeline updates the manuscript.
