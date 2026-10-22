"""Hardware-efficiency profiling: parameters, head parameters, size, FLOPs/MACs and latency."""

from __future__ import annotations

import io
import time
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

__all__ = [
    "count_parameters",
    "count_head_parameters",
    "model_size_mb",
    "count_flops",
    "measure_latency",
    "profile_model",
]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def count_head_parameters(model: nn.Module, head_module_names: Iterable[str]) -> int:
    total = 0
    for name in head_module_names:
        module = getattr(model, name, None)
        if module is not None:
            total += sum(p.numel() for p in module.parameters())
    return total


def model_size_mb(model: nn.Module) -> float:
    """Size of the serialised ``state_dict`` (float32 checkpoint) in megabytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024**2)


def _kan_basis_flops(model: nn.Module, x: torch.Tensor) -> int:
    """Analytic FLOPs of the element-wise Cox-de Boor recursion (not seen by the dispatcher)."""
    from src.models.kan_layers import KANConv2d, KANLinear

    counts = {"flops": 0}
    handles = []

    def make_hook(layer):
        def hook(_module, inputs, _output):
            inp = inputs[0]
            k = layer.spline_order
            n_knots = layer.grid_size + 2 * k + 1
            if isinstance(layer, KANLinear):
                n_elem = inp.numel()
            else:
                h_out, w_out = layer.output_size(inp.shape[-2], inp.shape[-1])
                n_elem = inp.shape[0] * layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1] * h_out * w_out
            # order-0 indicator (2 comparisons per interval) + 6 FLOPs per basis per order
            per_elem = 2 * (n_knots - 1) + sum(6 * (n_knots - 1 - p) for p in range(1, k + 1))
            per_elem += 4  # SiLU of the base branch
            counts["flops"] += int(n_elem * per_elem)

        return hook

    for module in model.modules():
        if isinstance(module, (KANLinear, KANConv2d)):
            handles.append(module.register_forward_hook(make_hook(module)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return counts["flops"]


def count_flops(model: nn.Module, input_size=(1, 3, 224, 224), device: Optional[torch.device] = None) -> Dict[str, float]:
    """FLOPs of one forward pass (matmul/conv/attention via ``FlopCounterMode`` + KAN bases)."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    x = torch.randn(*input_size, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    dense_flops = counter.get_total_flops()
    basis_flops = _kan_basis_flops(model, x)
    model.train(was_training)
    batch = input_size[0]
    total = (dense_flops + basis_flops) / batch
    return {
        "gflops": total / 1e9,
        "gmacs": dense_flops / batch / 2e9,
        "kan_basis_mflops": basis_flops / batch / 1e6,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_latency(
    model: nn.Module,
    input_size=(1, 3, 224, 224),
    warmup: int = 20,
    runs: int = 100,
    device: Optional[torch.device] = None,
    throughput_batch_size: int = 32,
    throughput_runs: int = 10,
) -> Dict[str, float]:
    """Single-image latency (ms, mean/std over ``runs`` after ``warmup``) and batched throughput."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    x = torch.randn(*input_size, device=device)
    timings = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        _sync(device)
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            _sync(device)
            timings.append((time.perf_counter() - start) * 1000.0)
        xb = torch.randn(throughput_batch_size, *input_size[1:], device=device)
        for _ in range(max(2, warmup // 5)):
            model(xb)
        _sync(device)
        start = time.perf_counter()
        for _ in range(throughput_runs):
            model(xb)
        _sync(device)
        elapsed = time.perf_counter() - start
    model.train(was_training)
    t = torch.tensor(timings)
    return {
        "latency_ms": float(t.mean()),
        "latency_std_ms": float(t.std()),
        "latency_median_ms": float(t.median()),
        "throughput_img_s": throughput_batch_size * throughput_runs / elapsed,
    }


def profile_model(
    model: nn.Module,
    head_module_names: Iterable[str],
    image_size: int = 224,
    device: Optional[torch.device] = None,
    warmup: int = 20,
    runs: int = 100,
    throughput_batch_size: int = 32,
) -> Dict[str, float]:
    """Complete hardware profile of ``model``."""
    device = device or next(model.parameters()).device
    model = model.to(device)
    total = count_parameters(model, trainable_only=True)
    head = count_head_parameters(model, head_module_names)
    result: Dict[str, float] = {
        "params_m": total / 1e6,
        "params_total": float(total),
        "head_params_k": head / 1e3,
        "head_params": float(head),
        "model_size_mb": model_size_mb(model),
    }
    result.update(count_flops(model, (1, 3, image_size, image_size), device))
    result.update(
        measure_latency(
            model, (1, 3, image_size, image_size), warmup=warmup, runs=runs,
            device=device, throughput_batch_size=throughput_batch_size,
        )
    )
    result["device"] = str(device)
    result["num_threads"] = torch.get_num_threads()
    return result
