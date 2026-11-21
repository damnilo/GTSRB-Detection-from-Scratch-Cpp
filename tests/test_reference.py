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

# Same order as Network::parameters(): Conv and Dense contribute weight then bias.
# Each BatchNorm contributes gamma then beta. Running statistics are buffers.
PARAMETER_NAMES = (
    "conv1.weight",
    "conv1.bias",
    "bn1.weight",
    "bn1.bias",
    "conv2.weight",
    "conv2.bias",
    "bn2.weight",
    "bn2.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
)
LAYER_NAMES = (
    "conv1",
    "bn1",
    "relu1",
    "pool1",
    "conv2",
    "bn2",
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
        self.bn1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward_layers(self, x):
        """One tensor per C++ layer, in Network order. Dropout follows train/eval mode."""
        layers = []
        x = self.conv1(x)
        layers.append(x)
        x = self.bn1(x)
        layers.append(x)
        x = F.relu(x)
        layers.append(x)
        x = self.pool(x)
        layers.append(x)

        x = self.conv2(x)
        layers.append(x)
        x = self.bn2(x)
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


def _cpp_to_torch_parameter(name, array):
    if name.startswith("conv") and name.endswith(".weight"):
        return _conv_cpp_to_torch(array)
    if name.startswith("fc") and name.endswith(".weight"):
        return _dense_cpp_to_torch(array)
    return torch.from_numpy(array.copy())


def _torch_to_cpp_array(name, tensor):
    if name.startswith("conv") and name.endswith(".weight"):
        return _torch_conv_to_cpp(tensor)
    if name.startswith("fc") and name.endswith(".weight"):
        return _torch_dense_to_cpp(tensor)
    return _as_numpy(tensor)


def _cpp_batch_norms(cpp_net):
    import cpp

    layers = []
    for index in range(cpp_net.numLayers()):
        layer = cpp_net.getLayer(index)
        if isinstance(layer, cpp.BatchNorm2D):
            layers.append(layer)
    return layers


def _copy_buffers_from_cpp(model, cpp_net):
    layers = _cpp_batch_norms(cpp_net)
    modules = (model.bn1, model.bn2)
    if len(layers) != len(modules):
        raise RuntimeError(f"expected {len(modules)} BatchNorm layers, found {len(layers)}")
    with torch.no_grad():
        for module, layer in zip(modules, layers):
            module.running_mean.copy_(torch.from_numpy(layer.runningMean().to_numpy().copy()))
            module.running_var.copy_(torch.from_numpy(layer.runningVar().to_numpy().copy()))


def _copy_buffers_to_cpp(model, cpp_net):
    layers = _cpp_batch_norms(cpp_net)
    modules = (model.bn1, model.bn2)
    for module, layer in zip(modules, layers):
        layer.runningMean().raw()[:] = _as_numpy(module.running_mean)
        layer.runningVar().raw()[:] = _as_numpy(module.running_var)


def copy_weights_from_cpp(model, cpp_net):
    """Load C++ parameters and BatchNorm running statistics into the PyTorch twin."""
    params = cpp_net.parameters()
    if len(params) != len(PARAMETER_NAMES):
        raise RuntimeError(f"expected {len(PARAMETER_NAMES)} C++ parameters, found {len(params)}")

    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, param in zip(PARAMETER_NAMES, params):
            array = param.to_numpy().astype(np.float32, copy=True)
            named[name].copy_(_cpp_to_torch_parameter(name, array))
    _copy_buffers_from_cpp(model, cpp_net)


def copy_weights_to_cpp(model, cpp_net):
    """Write PyTorch parameters and BatchNorm buffers back into the C++ network."""
    params = cpp_net.parameters()
    named = dict(model.named_parameters())
    for name, tensor in zip(PARAMETER_NAMES, params):
        tensor.raw()[:] = _torch_to_cpp_array(name, named[name]).reshape(-1)
    _copy_buffers_to_cpp(model, cpp_net)


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

    Dropout and BatchNorm are in eval mode on both sides, so masks and batch
    statistics do not enter the comparison. Running statistics are copied with the weights.
    Returns a list of (name, max_abs_error) and raises if any error exceeds atol.
    """
    from model import create_network as create_cpp_network
    from model import numpy_to_tensor, seed_everything, tensor_to_numpy

    seed_everything(seed)
    torch.manual_seed(seed)

    cpp_net, cpp_loss, _, cpp_dropout, cpp_batch_norms = create_cpp_network()
    model, loss, _, _ = create_network(num_classes)
    model.eval()
    cpp_dropout.setTraining(False)
    for layer in cpp_batch_norms:
        layer.setTraining(False)
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
    torch_grads = [_torch_to_cpp_array(name, named[name].grad) for name in PARAMETER_NAMES]
    for name, cpp_grad, torch_grad in zip(PARAMETER_NAMES, cpp_grads, torch_grads):
        rows.append((f"grad {name}", _max_abs(cpp_grad, torch_grad)))

    failures = [name for name, error in rows if error > atol]
    return rows, failures


def _grad_check(layer, values, eps=1e-3):
    """Central difference of sum(forward) against backward, one parameter at a time."""
    import cpp

    x = cpp.Tensor(np.asarray(values, dtype=np.float32))
    output = layer.forward(x)
    ones = np.ones(output.to_numpy().shape, dtype=np.float32)
    layer.backward(cpp.Tensor(ones))

    worst = 0.0
    for param, analytical in zip(layer.parameters(), layer.gradients()):
        data = param.raw()
        expected = analytical.to_numpy().reshape(-1)
        for index in range(data.size):
            original = float(data[index])
            data[index] = original + eps
            plus = float(layer.forward(x).to_numpy().sum())
            data[index] = original - eps
            minus = float(layer.forward(x).to_numpy().sum())
            data[index] = original
            numeric = (plus - minus) / (2.0 * eps)
            scale = max(1.0, abs(float(expected[index])))
            worst = max(worst, abs(numeric - float(expected[index])) / scale)
    return worst


def test_pytorch_reference_matches():
    _, failures = compare()
    assert not failures


def test_conv_finite_difference():
    import cpp

    from model import seed_everything

    seed_everything(1)
    layer = cpp.Conv2D(2, 3, 3, 1, 1)
    values = np.random.default_rng(1).standard_normal((2, 2, 5, 5)).astype(np.float32)
    assert _grad_check(layer, values) < 1e-3


def test_dense_finite_difference():
    import cpp

    from model import seed_everything

    seed_everything(0)
    layer = cpp.Dense(6, 4)
    values = np.random.default_rng(0).standard_normal((3, 6)).astype(np.float32)
    assert _grad_check(layer, values) < 1e-3


def test_softmax_rows_sum_to_one():
    import cpp

    from model import seed_everything

    seed_everything(0)
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-5.0, 5.0, 0.0]], dtype=np.float32)
    out = cpp.SoftMax().forward(cpp.Tensor(logits)).to_numpy()
    np.testing.assert_allclose(out.sum(axis=1), np.ones(3), atol=1e-6)
    assert np.all(out >= 0.0)


def test_dropout_eval_is_identity():
    import cpp

    from model import seed_everything

    seed_everything(0)
    layer = cpp.Dropout(0.9)
    layer.setTraining(False)
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = layer.forward(cpp.Tensor(values)).to_numpy()
    np.testing.assert_array_equal(out, values)


def test_serialization_roundtrip(tmp_path):
    import cpp

    from model import create_network, seed_everything

    seed_everything(7)
    net, _, _, _, _ = create_network(use_bn=False)
    before = [param.to_numpy().copy() for param in net.parameters()]
    path = tmp_path / "roundtrip.bin"
    cpp.Serialization.saveNetwork(net, str(path))

    seed_everything(99)
    restored, _, _, _, _ = create_network(use_bn=False)
    cpp.Serialization.loadNetwork(restored, str(path))
    for left, param in zip(before, restored.parameters()):
        np.testing.assert_array_equal(left, param.to_numpy())


def test_group_split_keeps_a_track_on_one_side():
    from load_dataset import stratified_group_split

    tracks = np.array(["t0a", "t0a", "t0b", "t0b", "t1a", "t1a", "t1b", "t1b"])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    train_idx, val_idx = stratified_group_split(tracks, labels, test_size=0.5, random_state=42)
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)

    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx) | set(val_idx) == set(range(len(labels)))
    for cls in np.unique(labels):
        train_tracks = set(tracks[train_idx][labels[train_idx] == cls])
        val_tracks = set(tracks[val_idx][labels[val_idx] == cls])
        assert train_tracks.isdisjoint(val_tracks)
        assert train_tracks and val_tracks


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
