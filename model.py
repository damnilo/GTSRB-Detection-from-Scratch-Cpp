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
BN_PATH = CHECKPOINT_DIR / "bn.npz"
CONFIG_PATH = CHECKPOINT_DIR / "config.json"

EPOCHS = 10
BATCH_SIZE = 64
SEED = 42
LR = 0.015
MOMENTUM = 0.95
WEIGHT_DECAY = 1e-4
ADAMW_LR = 1e-3
ADAMW_WEIGHT_DECAY = 1e-2
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

# OpenBLAS and the tiled kernel share this pool. It has to be set before the
# module loads. Four threads stay on the performance cores; using every logical
# processor makes the convolution weight gradient wait on the slow cores.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2
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


def save_norm(mean, std, path=NORM_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), mean=np.asarray(mean, dtype=np.float32), std=np.asarray(std, dtype=np.float32))


def save_config(directory=CHECKPOINT_DIR, seed=SEED, use_aug=True, use_bn=True, optimizer_name="sgd"):
    directory = Path(directory)
    norm_layer = "BatchNorm, " if use_bn else ""
    if optimizer_name == "adamw":
        recorded_lr = ADAMW_LR
        recorded_decay = ADAMW_WEIGHT_DECAY
    else:
        recorded_lr = LR
        recorded_decay = WEIGHT_DECAY
    config = {
        "architecture": (
            f"Conv 3->16 k3 s1 p1, {norm_layer}ReLU, MaxPool 2; "
            f"Conv 16->32 k3 s1 p1, {norm_layer}ReLU, MaxPool 2; "
            "Flatten; Dense 2048->128; ReLU; Dropout 0.3; "
            "Dense 128->43; SoftMax"
        ),
        "optimizer": optimizer_name,
        "lr": recorded_lr,
        "momentum": MOMENTUM,
        "weight_decay": recorded_decay,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": seed,
        "use_aug": use_aug,
        "use_bn": use_bn,
        "clip_value": CLIP,
        "image_size": 32,
        "num_classes": NUM_CATEGORIES,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

def augment_image(chw):
    """Geometric and photometric jitter on one CHW image in [0, 1]."""
    hwc = np.ascontiguousarray(np.moveaxis(chw, 0, -1))
    angle = np.random.uniform(-15.0, 15.0)
    scale = np.random.uniform(0.9, 1.1)
    shift_x = np.random.uniform(-2.0, 2.0)
    shift_y = np.random.uniform(-2.0, 2.0)

    height, width = hwc.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    hwc = cv2.warpAffine(hwc, matrix, (width, height))

    if np.random.random() < 0.5:
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-0.1, 0.1)
        hwc = alpha * hwc + beta

    if np.random.random() < 0.3:
        hwc = cv2.GaussianBlur(hwc, (3, 3), 0)

    hwc = np.clip(hwc, 0.0, 1.0)
    return np.moveaxis(hwc, -1, 0)


def augment_batch(batch):
    """Augment each NCHW image. The caller standardizes the result."""
    return np.stack([augment_image(image) for image in batch])

def load_norm_stats(path=None):
    """Load training-set mean/std. If the default file is missing, recompute it from the train split.

    Recomputing matches an existing best_model.bin only when os.listdir returns
    training images in the same order as the run that produced those weights.
    """
    path = NORM_PATH if path is None else Path(path)
    if not path.is_file():
        if path != NORM_PATH:
            raise FileNotFoundError(f"{path} is missing. This run has no saved training-set mean/std.")
        print("checkpoints/norm.npz was not found. Recomputing mean/std from the training split.")
        print("This matches best_model.bin only if training files are listed in the same order as when that checkpoint was trained.")
        X_train, _, _, _ = load_train_data()
        X_train = X_train.astype(np.float32)
        mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
        std = np.std(X_train, axis=(0, 2, 3), keepdims=True)
        save_norm(mean, std)
        return mean, std

    data = np.load(path)
    return data["mean"], data["std"]


def numpy_to_tensor(np_array):
    """Wraps a numpy array as a cpp.Tensor (copies data into the C++ buffer)."""
    return cpp.Tensor(np_array.astype(np.float32))


def tensor_to_numpy(tensor):
    """Copies a cpp.Tensor's contents into a fresh numpy array."""
    return tensor.to_numpy().astype(np.float32)


def create_network(use_bn=True, optimizer_name="sgd"):
    """
    Builds the model: two conv blocks, a Flatten, Dense, Dropout, and SoftMax.

    BatchNorm sits after each convolution when use_bn is set. The dropout layer
    and any batch-norm layers are returned so the training loop can switch them
    between train and eval mode.
    """
    net = cpp.Network()
    batch_norms = []

    net.addLayer(cpp.Conv2D(3, 16, 3, 1, 1))
    if use_bn:
        batch_norm1 = cpp.BatchNorm2D(16)
        net.addLayer(batch_norm1)
        batch_norms.append(batch_norm1)
    net.addLayer(cpp.ReLU())
    net.addLayer(cpp.MaxPool2D(2, 2))

    net.addLayer(cpp.Conv2D(16, 32, 3, 1, 1))
    if use_bn:
        batch_norm2 = cpp.BatchNorm2D(32)
        net.addLayer(batch_norm2)
        batch_norms.append(batch_norm2)
    net.addLayer(cpp.ReLU())
    net.addLayer(cpp.MaxPool2D(2, 2))

    net.addLayer(cpp.Flatten())
    net.addLayer(cpp.Dense(32 * 8 * 8, 128))
    net.addLayer(cpp.ReLU())

    dropout_layer = cpp.Dropout(0.3)
    net.addLayer(dropout_layer)

    net.addLayer(cpp.Dense(128, NUM_CATEGORIES))
    net.addLayer(cpp.SoftMax())

    loss = cpp.CrossEntropy()
    if optimizer_name == "adamw":
        optimizer = cpp.AdamW(ADAMW_LR, ADAMW_WEIGHT_DECAY)
    elif optimizer_name == "sgd":
        optimizer = cpp.SGD(LR, MOMENTUM)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    return net, loss, optimizer, dropout_layer, tuple(batch_norms)


def set_training(dropout, batch_norms, is_training):
    """Train mode uses dropout masks and batch statistics. Eval mode uses running statistics."""
    dropout.setTraining(is_training)
    for layer in batch_norms:
        layer.setTraining(is_training)


def save_batch_norm(batch_norms, path=BN_PATH):
    """Save running mean and variance. They are not part of best_model.bin."""
    if not batch_norms:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index, layer in enumerate(batch_norms):
        arrays[f"mean_{index}"] = np.asarray(layer.runningMean().to_numpy(), dtype=np.float32)
        arrays[f"var_{index}"] = np.asarray(layer.runningVar().to_numpy(), dtype=np.float32)
    np.savez(str(path), **arrays)


def load_batch_norm(batch_norms, path=BN_PATH):
    """Load the running statistics written beside the checkpoint these layers belong to."""
    if not batch_norms:
        return
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{path} is missing. Evaluation needs the running mean and variance saved with best_model.bin."
        )
    data = np.load(path)
    for index, layer in enumerate(batch_norms):
        layer.runningMean().raw()[:] = np.asarray(data[f"mean_{index}"], dtype=np.float32)
        layer.runningVar().raw()[:] = np.asarray(data[f"var_{index}"], dtype=np.float32)


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


def train(seed=SEED, use_aug=True, use_bn=True, checkpoint_dir=None, optimizer_name="sgd"):
    """
    Main training loop. Key design points:
      - Mean/std come from the unaugmented training set. Val is standardized
        once with those stats and is never augmented.
      - use_aug applies the train-batch jitter before that standardization.
      - use_bn inserts BatchNorm after each convolution. Its running statistics
        are saved beside the best checkpoint; gamma and beta are in the weights.
      - Weight init, dropout masks, and the NumPy oversampling draw are seeded.
      - Class-imbalance is addressed via sqrt-dampened inverse-frequency
        oversampling (capped at 3x the median weight).
      - Dropout and BatchNorm are on for training batches and off before
        validation. Learning rate is halved when validation accuracy does not
        improve, and the best model is written after every improving epoch.
    """
    directory = CHECKPOINT_DIR if checkpoint_dir is None else Path(checkpoint_dir)
    model_path = "best_model.bin" if checkpoint_dir is None else str(directory / "best_model.bin")
    seed_everything(seed)

    X_train, y_train, X_val, y_val = load_train_data()
    y_train_one_hot = np.eye(NUM_CATEGORIES)[y_train]

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    std = np.std(X_train, axis=(0, 2, 3), keepdims=True)
    save_norm(mean, std, directory / "norm.npz")
    save_config(directory, seed, use_aug, use_bn, optimizer_name)

    X_val = standardize(X_val, mean, std)

    net, loss, optimizer, dropout_layer, batch_norms = create_network(use_bn=use_bn, optimizer_name=optimizer_name)

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
        set_training(dropout_layer, batch_norms, True)

        for i in range(0, len(X_train), BATCH_SIZE):
            if (i // BATCH_SIZE) % 40 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {i//BATCH_SIZE + 1}/{len(X_train)//BATCH_SIZE}")

            batch_indices = permutation[i:i+BATCH_SIZE]

            images = X_train[batch_indices]
            if use_aug:
                images = augment_batch(images)
            X_batch = standardize(images, mean, std)
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

        set_training(dropout_layer, batch_norms, False)
        avg_loss = epoch_loss / batches
        val_acc = evaluate(net, X_val, y_val)
        train_check_acc = evaluate(net, standardize(X_train[:2000], mean, std), y_train[:2000])
        print("Train accuracy:", train_check_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Accuracy: {val_acc * 100:.2f}%, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            cpp.Serialization.saveNetwork(net, model_path)
            if batch_norms:
                save_batch_norm(batch_norms, directory / "bn.npz")
        else:
            optimizer.setLearningRate(optimizer.getLearningRate() * 0.5)

    return best_acc


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
