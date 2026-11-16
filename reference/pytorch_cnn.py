"""PyTorch twin of the C++ network, plus a numeric comparison against it.

The C++ model ends in a SoftMax layer and a cross-entropy that expects
probabilities, not logits. nn.CrossEntropyLoss would apply a second softmax,
so the loss here is the same mean -sum(t log p) used in CrossEntropy.cpp.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LR = 0.015
MOMENTUM = 0.95
WEIGHT_DECAY = 1e-4

# Same order as Network::parameters(): each Conv/Dense contributes weight, then bias.
PARAMETER_NAMES = (
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
)
LAYER_NAMES = (
    "conv1",
    "relu1",
    "pool1",
    "conv2",
    "relu2",
    "pool2",
    "flatten",
    "fc1",
    "relu3",
    "dropout",
    "fc2",
    "softmax",
)


class CNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward_layers(self, x):
        """One tensor per C++ layer, in Network order. Dropout follows train/eval mode."""
        layers = []
        x = self.conv1(x)
        layers.append(x)
        x = F.relu(x)
        layers.append(x)
        x = self.pool(x)
        layers.append(x)

        x = self.conv2(x)
        layers.append(x)
        x = F.relu(x)
        layers.append(x)
        x = self.pool(x)
        layers.append(x)

        x = torch.flatten(x, 1)
        layers.append(x)
        x = self.fc1(x)
        layers.append(x)
        x = F.relu(x)
        layers.append(x)
        x = self.dropout(x)
        layers.append(x)
        x = self.fc2(x)
        layers.append(x)
        x = F.softmax(x, dim=1)
        layers.append(x)
        return layers

    def forward(self, x):
        return self.forward_layers(x)[-1]


class ProbabilityCrossEntropy(nn.Module):
    """C++ CrossEntropy: batch mean of -sum_c t log(max(p, eps)), p already a probability."""

    def forward(self, predictions, targets):
        eps = 1e-12
        active = targets > 0
        log_p = predictions.clamp(min=eps).log()
        return -(targets[active] * log_p[active]).sum() / predictions.shape[0]


def create_network(num_classes=43):
    model = CNN(num_classes)
    loss = ProbabilityCrossEntropy()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    return model, loss, optimizer, model.dropout


def _as_numpy(tensor):
    return tensor.detach().cpu().contiguous().numpy()


def _conv_cpp_to_torch(weight):
    # C++ (out, in*k*k), channel then kernel-row then kernel-col. PyTorch is (out, in, k, k).
    out_channels = weight.shape[0]
    kernel = 3
    in_channels = weight.shape[1] // (kernel * kernel)
    return torch.from_numpy(weight.reshape(out_channels, in_channels, kernel, kernel).copy())


def _dense_cpp_to_torch(weight):
    # C++ (in, out) is the transpose of Linear.weight (out, in).
    return torch.from_numpy(weight.T.copy())


def _torch_conv_to_cpp(weight):
    out_channels = weight.shape[0]
    return _as_numpy(weight).reshape(out_channels, -1)


def _torch_dense_to_cpp(weight):
    return _as_numpy(weight.T)


def copy_weights_from_cpp(model, cpp_net):
    """Load the eight C++ parameter tensors into the matching PyTorch parameters."""
    params = cpp_net.parameters()
    if len(params) != len(PARAMETER_NAMES):
        raise RuntimeError(f"expected {len(PARAMETER_NAMES)} C++ parameters, found {len(params)}")

    numpy = [params[i].to_numpy().astype(np.float32, copy=True) for i in range(len(params))]
    mapping = {
        "conv1.weight": _conv_cpp_to_torch(numpy[0]),
        "conv1.bias": torch.from_numpy(numpy[1].copy()),
        "conv2.weight": _conv_cpp_to_torch(numpy[2]),
        "conv2.bias": torch.from_numpy(numpy[3].copy()),
        "fc1.weight": _dense_cpp_to_torch(numpy[4]),
        "fc1.bias": torch.from_numpy(numpy[5].copy()),
        "fc2.weight": _dense_cpp_to_torch(numpy[6]),
        "fc2.bias": torch.from_numpy(numpy[7].copy()),
    }
    with torch.no_grad():
        for name, value in mapping.items():
            parameter = dict(model.named_parameters())[name]
            parameter.copy_(value)


def copy_weights_to_cpp(model, cpp_net):
    """Write PyTorch parameters back into the C++ network, in parameters() order."""
    params = cpp_net.parameters()
    named = dict(model.named_parameters())
    arrays = [
        _torch_conv_to_cpp(named["conv1.weight"]),
        _as_numpy(named["conv1.bias"]),
        _torch_conv_to_cpp(named["conv2.weight"]),
        _as_numpy(named["conv2.bias"]),
        _torch_dense_to_cpp(named["fc1.weight"]),
        _as_numpy(named["fc1.bias"]),
        _torch_dense_to_cpp(named["fc2.weight"]),
        _as_numpy(named["fc2.bias"]),
    ]
    for tensor, array in zip(params, arrays):
        tensor.raw()[:] = array.reshape(-1)


def _cpp_activations(cpp_net, x_np, numpy_to_tensor, tensor_to_numpy):
    current = numpy_to_tensor(x_np)
    activations = []
    for index in range(cpp_net.numLayers()):
        current = cpp_net.getLayer(index).forward(current)
        activations.append(tensor_to_numpy(current))
    return activations


def _max_abs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def compare(batch_size=4, num_classes=43, atol=1e-4, seed=0):
    """Copy one C++ initialization into PyTorch and report max |Δ| for forward and backward.

    Dropout is in eval mode on both sides, so the masks do not enter the comparison.
    Returns a list of (name, max_abs_error) and raises if any error exceeds atol.
    """
    from model import create_network as create_cpp_network
    from model import numpy_to_tensor, seed_everything, tensor_to_numpy

    seed_everything(seed)
    torch.manual_seed(seed)

    cpp_net, cpp_loss, _, cpp_dropout = create_cpp_network()
    model, loss, _, _ = create_network(num_classes)
    model.eval()
    cpp_dropout.setTraining(False)
    copy_weights_from_cpp(model, cpp_net)

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((batch_size, 3, 32, 32), dtype=np.float32)
    labels = rng.integers(0, num_classes, size=batch_size)
    y_np = np.eye(num_classes, dtype=np.float32)[labels]

    cpp_layers = _cpp_activations(cpp_net, x_np, numpy_to_tensor, tensor_to_numpy)
    torch_layers = model.forward_layers(torch.from_numpy(x_np))

    rows = []
    for name, cpp_act, torch_act in zip(LAYER_NAMES, cpp_layers, torch_layers):
        rows.append((f"forward {name}", _max_abs(cpp_act, _as_numpy(torch_act))))

    y_torch = torch.from_numpy(y_np)
    torch_loss = loss(torch_layers[-1], y_torch)
    cpp_loss_value = cpp_loss.forward(numpy_to_tensor(cpp_layers[-1]), numpy_to_tensor(y_np))
    rows.append(("loss", abs(float(torch_loss.detach()) - float(cpp_loss_value))))

    cpp_net.backward(cpp_loss.backward())
    torch_loss.backward()

    cpp_grads = [tensor_to_numpy(grad) for grad in cpp_net.gradients()]
    named = dict(model.named_parameters())
    torch_grads = [
        _torch_conv_to_cpp(named["conv1.weight"].grad),
        _as_numpy(named["conv1.bias"].grad),
        _torch_conv_to_cpp(named["conv2.weight"].grad),
        _as_numpy(named["conv2.bias"].grad),
        _torch_dense_to_cpp(named["fc1.weight"].grad),
        _as_numpy(named["fc1.bias"].grad),
        _torch_dense_to_cpp(named["fc2.weight"].grad),
        _as_numpy(named["fc2.bias"].grad),
    ]
    for name, cpp_grad, torch_grad in zip(PARAMETER_NAMES, cpp_grads, torch_grads):
        rows.append((f"grad {name}", _max_abs(cpp_grad, torch_grad)))

    failures = [name for name, error in rows if error > atol]
    return rows, failures


def main():
    rows, failures = compare()
    for name, error in rows:
        print(f"{name:22s}  max |Δ| = {error:.3e}")
    if failures:
        print("Above 1e-4:", ", ".join(failures))
        raise SystemExit(1)
    print("All comparisons are within 1e-4.")


if __name__ == "__main__":
    main()
