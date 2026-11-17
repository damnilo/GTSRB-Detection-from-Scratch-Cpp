"""CPU timing of the C++ network against the PyTorch reference.

Measures full eval, each Conv forward, and each Dense forward at batch 1/16/64.
Inputs are built once, outside the timer. Dropout is off on both sides.
Set this before the C++ module loads: OpenMP reads OMP_NUM_THREADS at startup.
"""

import os

THREADS = 4
os.environ["OMP_NUM_THREADS"] = str(THREADS)

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from model import create_network as create_cpp_network
from model import numpy_to_tensor
from reference.pytorch_cnn import copy_weights_from_cpp
from reference.pytorch_cnn import create_network as create_torch_network
import cpp

BATCHES = [1, 16, 64]
WARMUP = 5
REPEATS = 10

# Matmul inside each layer, per image: (M, K, N). Conv M is the spatial size.
GEMM_PER_IMAGE = {
    "conv1": (32 * 32, 27, 16),
    "conv2": (16 * 16, 144, 32),
    "fc1": (1, 2048, 128),
    "fc2": (1, 128, 43),
}


def time_median(func):
    for _ in range(WARMUP):
        func()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        func()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def gflops(name, batch, seconds):
    spatial, k, n = GEMM_PER_IMAGE[name]
    m = spatial * batch
    return (2.0 * m * k * n) / (seconds * 1e9)


def layer_inputs(batch):
    rng = np.random.default_rng(0)
    return {
        "conv1": rng.standard_normal((batch, 3, 32, 32), dtype=np.float32),
        "conv2": rng.standard_normal((batch, 16, 16, 16), dtype=np.float32),
        "fc1": rng.standard_normal((batch, 2048), dtype=np.float32),
        "fc2": rng.standard_normal((batch, 128), dtype=np.float32),
    }


def bench_pair(cpp_fn, torch_fn):
    cpp_s = time_median(cpp_fn)
    torch_s = time_median(torch_fn)
    return cpp_s * 1000.0, torch_s * 1000.0


def bench():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    cpp_net, _, _, dropout, batch_norms = create_cpp_network()
    torch_net, _, _, _ = create_torch_network()
    copy_weights_from_cpp(torch_net, cpp_net)
    dropout.setTraining(False)
    for layer in batch_norms:
        layer.setTraining(False)
    torch_net.eval()

    cpp_layers = {
        "conv1": cpp.Conv2D(3, 16, 3, 1, 1),
        "conv2": cpp.Conv2D(16, 32, 3, 1, 1),
        "fc1": cpp.Dense(2048, 128),
        "fc2": cpp.Dense(128, 43),
    }
    torch_layers = {
        "conv1": torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        "conv2": torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        "fc1": torch.nn.Linear(2048, 128),
        "fc2": torch.nn.Linear(128, 43),
    }

    print(f"threads={THREADS}  warmup={WARMUP}  repeats={REPEATS}  median")
    print(f"{'case':<12} {'batch':>5} {'cpp ms':>10} {'torch ms':>10} {'cpp img/s':>10} {'torch img/s':>12} {'cpp GFLOP/s':>12}")

    for batch in BATCHES:
        full = layer_inputs(batch)["conv1"]
        cpp_x = numpy_to_tensor(full)
        torch_x = torch.from_numpy(full)
        cpp_ms, torch_ms = bench_pair(
            lambda: cpp_net.forward(cpp_x),
            lambda: torch_net(torch_x),
        )
        flops = sum(gflops(name, batch, cpp_ms / 1000.0) for name in GEMM_PER_IMAGE)
        _report("full", batch, cpp_ms, torch_ms, flops)

        inputs = layer_inputs(batch)
        for name in ("conv1", "conv2", "fc1", "fc2"):
            raw = inputs[name]
            cpp_in = numpy_to_tensor(raw)
            torch_in = torch.from_numpy(raw)
            cpp_layer = cpp_layers[name]
            torch_layer = torch_layers[name]
            cpp_ms, torch_ms = bench_pair(
                lambda layer=cpp_layer, tensor=cpp_in: layer.forward(tensor),
                lambda layer=torch_layer, tensor=torch_in: layer(tensor),
            )
            _report(name, batch, cpp_ms, torch_ms, gflops(name, batch, cpp_ms / 1000.0))


def _report(name, batch, cpp_ms, torch_ms, cpp_gflops):
    print(
        f"{name:<12} {batch:5d} {cpp_ms:10.3f} {torch_ms:10.3f} "
        f"{batch / (cpp_ms / 1000.0):10.1f} {batch / (torch_ms / 1000.0):12.1f} "
        f"{cpp_gflops:12.2f}"
    )


if __name__ == "__main__":
    bench()
