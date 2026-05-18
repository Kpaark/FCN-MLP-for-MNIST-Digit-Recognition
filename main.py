import argparse
from pathlib import Path

import numpy as np


def load_mnist(npz_path: str):
    data = np.load(npz_path)
    X = data["image"].astype(np.float32) / 255.0
    y = data["label"].astype(np.int64)
    return X, y


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def parse_args():
    p = argparse.ArgumentParser(description="NumPy MLP on MNIST")
    p.add_argument("--data", type=str, default="mnist.npz", help="MNIST .npz path")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"missing {data_path}. put mnist.npz in folder"
        )

    X, y = load_mnist(str(data_path))
    X_flat = flatten(X)
    print(f"loaded {len(y)} images, shape {X_flat.shape}, labels 0–9")


if __name__ == "__main__":
    main()
