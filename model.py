import json
import os
import random
import sys
import time
from pathlib import Path

from load_dataset import NUM_CATEGORIES, load_train_data


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
NORM_PATH = CHECKPOINT_DIR / "norm.npz"
CONFIG_PATH = CHECKPOINT_DIR / "config.json"

EPOCHS = 20
BATCH_SIZE = 64
SEED = 42
LR = 0.015
MOMENTUM = 0.95
WEIGHT_DECAY = 1e-4
CLIP = 5.0


def add_cpp_to_path():
    """Put the local build directory on the module path before importing cpp.

    On Windows, OpenMP lives in the MSYS2 toolchain. Set MSYS2_BIN to that
    bin directory when it is not C:\\msys64\\ucrt64\\bin and not already on PATH.
    The project build itself is always <repo>/build.
    """
    build = ROOT / "build"
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        if build.is_dir():
            os.add_dll_directory(str(build))
        msys = os.environ.get("MSYS2_BIN")
        if not msys:
            default_msys = Path(r"C:\msys64\ucrt64\bin")
            if default_msys.is_dir():
                msys = str(default_msys)
        if msys:
            os.add_dll_directory(msys)
    sys.path.insert(0, str(build))


add_cpp_to_path()

import numpy as np
import cpp


def seed_everything(seed):
    """Seed Python, NumPy, weight init, and dropout before the network is built."""
    random.seed(seed)
    np.random.seed(seed)
    cpp.seed(seed)


def standardize(X, mean, std):
    """Per-channel standardization. mean/std broadcast over NCHW as (1, C, 1, 1)."""
    return (X - mean) / std


def save_norm(mean, std):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(str(NORM_PATH), mean=np.asarray(mean, dtype=np.float32), std=np.asarray(std, dtype=np.float32))


def save_config():
    config = {
        "architecture": (
            "Conv 3->16 k3 s1 p1, ReLU, MaxPool 2; "
            "Conv 16->32 k3 s1 p1, ReLU, MaxPool 2; "
            "Flatten; Dense 2048->128; ReLU; Dropout 0.3; "
            "Dense 128->43; SoftMax"
        ),
        "lr": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "clip_value": CLIP,
        "image_size": 32,
        "num_classes": NUM_CATEGORIES,
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_norm_stats():
    """Load training-set mean/std. If the file is missing, recompute it from the train split.

    Recomputing matches an existing best_model.bin only when os.listdir returns
    training images in the same order as the run that produced those weights.
    """
    if not NORM_PATH.is_file():
        print("checkpoints/norm.npz was not found. Recomputing mean/std from the training split.")
        print("This matches best_model.bin only if training files are listed in the same order as when that checkpoint was trained.")
        X_train, _, _, _ = load_train_data()
        X_train = X_train.astype(np.float32)
        mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
        std = np.std(X_train, axis=(0, 2, 3), keepdims=True)
        save_norm(mean, std)
        return mean, std

    data = np.load(NORM_PATH)
    return data["mean"], data["std"]


def numpy_to_tensor(np_array):
    """Wraps a numpy array as a cpp.Tensor (copies data into the C++ buffer)."""
    return cpp.Tensor(np_array.astype(np.float32))


def tensor_to_numpy(tensor):
    """Copies a cpp.Tensor's contents into a fresh numpy array."""
    return tensor.to_numpy().astype(np.float32)


def create_network():
    """
    Builds the model architecture: two conv blocks (each Conv2D -> ReLU -> MaxPool2D),
    a Flatten, a Dense hidden layer with Dropout regularization, and a final
    Dense -> SoftMax classification head over NUM_CATEGORIES classes.

    Returns the network, loss function, optimizer, and a direct reference to the
    dropout layer (needed to toggle train/eval mode from the training loop).
    """
    net = cpp.Network()

    net.addLayer(cpp.Conv2D(3, 16, 3, 1, 1))
    net.addLayer(cpp.ReLU())
    net.addLayer(cpp.MaxPool2D(2, 2))

    net.addLayer(cpp.Conv2D(16, 32, 3, 1, 1))
    net.addLayer(cpp.ReLU())
    net.addLayer(cpp.MaxPool2D(2, 2))

    net.addLayer(cpp.Flatten())
    net.addLayer(cpp.Dense(32 * 8 * 8, 128))
    net.addLayer(cpp.ReLU())

    dropout_layer = cpp.Dropout(0.3);
    net.addLayer(dropout_layer)

    net.addLayer(cpp.Dense(128, NUM_CATEGORIES))
    net.addLayer(cpp.SoftMax())

    loss = cpp.CrossEntropy()
    optimizer = cpp.SGD(LR, MOMENTUM)

    return net, loss, optimizer, dropout_layer


def clip_gradients(grads, clip_value=1.0):
    """
    Global-norm gradient clipping: scales ALL gradient tensors uniformly by a
    single factor if their combined L2 norm exceeds clip_value. This preserves
    the gradient's overall direction, unlike per-tensor clipping which would
    distort the relative scale between layers (especially harmful to the large
    Dense layer's gradients).

    Note: uses to_numpy() (a copy) just for the norm calculation, but raw()
    (a zero-copy view) for the actual in-place scaling — clip_gradients only
    works because raw() writes straight back into the Tensor's own buffer.
    """
    total_norm = np.sqrt(sum(np.sum(g.to_numpy() ** 2) for g in grads))
    if total_norm > clip_value:
        scale = clip_value / (total_norm + 1e-6)
        for g in grads:
            g.raw()[:] *= scale


def train():
    """
    Main training loop. Key design points:
      - Inputs are normalized using per-channel mean/std computed from the
        training set only (val set reuses train's statistics to avoid leakage).
        Those statistics are written to checkpoints/norm.npz before training.
      - Weight init, dropout masks, and the NumPy oversampling draw are seeded.
      - Class-imbalance is addressed via sqrt-dampened inverse-frequency
        oversampling (capped at 3x the median weight) — this softens the more
        aggressive raw 1/count weighting, which was found to over-correct rare
        classes at the expense of nearby ones.
      - Dropout is explicitly toggled on for training batches and off before
        each evaluation call, since eval-mode Dropout must be a pure pass-through.
      - Learning rate is halved whenever an epoch fails to improve validation
        accuracy (a simple reduce-on-plateau schedule); the best model (by val
        accuracy) is checkpointed to disk after every improving epoch.
    """
    seed_everything(SEED)

    X_train, y_train, X_val, y_val = load_train_data()
    y_train_one_hot = np.eye(NUM_CATEGORIES)[y_train]

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    std = np.std(X_train, axis=(0, 2, 3), keepdims=True)
    save_norm(mean, std)
    save_config()

    X_train = standardize(X_train, mean, std)
    X_val = standardize(X_val, mean, std)

    net, loss, optimizer, dropout_layer = create_network()

    params = net.parameters()
    grads = net.gradients()
    start_time = time.time()
    best_acc = 0.0

    class_counts = np.bincount(y_train, minlength=NUM_CATEGORIES)
    class_weights = 1.0 / np.sqrt(class_counts)
    class_weights = np.minimum(class_weights, np.median(class_weights) * 3)
    sample_weights = class_weights[y_train]
    sample_weights /= sample_weights.sum()

    for epoch in range(EPOCHS):
        permutation = np.random.choice(len(X_train), size=len(X_train), replace=True, p=sample_weights)
        epoch_loss = 0.0
        batches = 0
        dropout_layer.setTraining(True)

        for i in range(0, len(X_train), BATCH_SIZE):
            if (i // BATCH_SIZE) % 40 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {i//BATCH_SIZE + 1}/{len(X_train)//BATCH_SIZE}")

            batch_indices = permutation[i:i+BATCH_SIZE]

            X_batch = X_train[batch_indices]
            y_batch = y_train_one_hot[batch_indices]

            X_tensor = numpy_to_tensor(X_batch)
            y_tensor = numpy_to_tensor(y_batch)

            predictions = net.forward(X_tensor)
            loss_value = loss.forward(predictions, y_tensor)
            epoch_loss += loss_value
            batches += 1

            grad_loss = loss.backward()
            net.backward(grad_loss)
            clip_gradients(grads, clip_value=CLIP)
            optimizer.update(params, grads)

        dropout_layer.setTraining(False)
        avg_loss = epoch_loss / batches
        val_acc = evaluate(net, X_val, y_val)
        train_check_acc = evaluate(net, X_train[:2000], y_train[:2000])
        print("Train accuracy:", train_check_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Accuracy: {val_acc * 100:.2f}%, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            cpp.Serialization.saveNetwork(net, "best_model.bin")
        else:
            optimizer.setLearningRate(optimizer.getLearningRate() * 0.5)


def evaluate(net, X_val, y_val):
    """Computes classification accuracy of `net` on (X_val, y_val), batched to bound memory use."""
    correct = 0
    batch_size = 64

    for i in range(0, len(X_val), batch_size):
        if (i // batch_size) % 40 == 0:
            print(f"Evaluating batch {i//batch_size + 1}/{len(X_val)//batch_size}")
        X_batch = X_val[i:i+batch_size]
        y_batch = y_val[i:i+batch_size]

        X_tensor = numpy_to_tensor(X_batch)
        predictions = tensor_to_numpy(net.forward(X_tensor))

        labels_batch = np.argmax(predictions, axis=1)
        batch_correct = np.sum(labels_batch == y_batch)

        correct += batch_correct

    return correct / len(X_val)


def evaluate_with_confusion(net, X_val, y_val, num_classes):
    """
    Same forward pass as evaluate(), but accumulates a full confusion matrix
    (rows = true class, columns = predicted class) instead of just an accuracy
    scalar — used to diagnose which specific classes/class-pairs the model
    struggles with, beyond what a single overall accuracy number reveals.
    """
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(0, len(X_val), 64):
        X_batch = X_val[i:i+64]
        y_batch = y_val[i:i+64]
        preds = np.argmax(tensor_to_numpy(net.forward(numpy_to_tensor(X_batch))), axis=1)
        for t, p in zip(y_batch, preds):
            confusion[t, p] += 1
    return confusion


if __name__ == "__main__":
    train()
