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


class MLP:
    def __init__(self, hidden_size: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / 784), (784, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_size), (hidden_size, 10)).astype(np.float32)
        self.b2 = np.zeros(10, dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    def backward(self, X: np.ndarray, y_oh: np.ndarray, lr: float) -> None:
        batch = X.shape[0]
        dz2 = (self.probs - y_oh) / batch
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)


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
