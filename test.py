"""
Loads a trained CNN and evaluates it on the official GTSRB test set.

Images are standardized with the training-set per-channel mean and standard
deviation stored in checkpoints/norm.npz, not with statistics of the test set.
"""

import os

import cv2
import numpy as np
import pandas as pd

from load_dataset import IMG_HEIGHT, IMG_WIDTH
from model import SEED, create_network, evaluate, load_norm_stats, seed_everything, standardize

import cpp


def load_test_data():
    """
    Loads and preprocesses the test images and labels.

    Pixel values are scaled to [0, 1] and stored as NCHW. Mean/std
    standardization is intentionally not applied here.

    Returns:
        tuple: Test images and corresponding labels.
    """

    csv_path = os.path.join("dataset", "Test.csv")
    df = pd.read_csv(csv_path)
    image_data = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join("dataset", row["Path"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        image_data.append(img)
        labels.append(row["ClassId"])

    image_data = np.array(image_data, dtype=np.float32)
    image_data = image_data.transpose(0, 3, 1, 2) / 255.0
    labels = np.array(labels, dtype=np.int32)

    return image_data, labels


def main():
    """Load best_model.bin and print official-test accuracy under training-set normalization."""
    seed_everything(SEED)
    net, _, _, dropout = create_network()
    cpp.Serialization.loadNetwork(net, "best_model.bin")
    dropout.setTraining(False)

    X_test, y_test = load_test_data()
    mean, std = load_norm_stats()
    X_test = standardize(X_test, mean, std)

    test_acc = evaluate(net, X_test, y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
