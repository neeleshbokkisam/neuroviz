import numpy as np
from sklearn.datasets import make_moons, make_circles


def generate_xor_dataset(n_samples=200, noise=0.1, random_state=42):
    np.random.seed(random_state)

    n = n_samples // 4
    X1 = np.random.randn(n, 2) * noise + np.array([0, 0])
    X2 = np.random.randn(n, 2) * noise + np.array([1, 1])
    X3 = np.random.randn(n, 2) * noise + np.array([0, 1])
    X4 = np.random.randn(n, 2) * noise + np.array([1, 0])

    X = np.vstack([X1, X2, X3, X4])
    y = np.hstack([np.zeros(n), np.zeros(n), np.ones(n), np.ones(n)])

    indices = np.random.permutation(len(X))
    return X[indices], y[indices].reshape(-1, 1)


def generate_moons_dataset(n_samples=200, noise=0.1, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y.reshape(-1, 1)


def generate_circles_dataset(n_samples=200, noise=0.1, random_state=42):
    X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state, factor=0.5)
    return X, y.reshape(-1, 1)


def load_dataset(dataset_name, n_samples=200, noise=0.1, random_state=42):
    if dataset_name == 'xor':
        return generate_xor_dataset(n_samples, noise, random_state)
    elif dataset_name == 'moons':
        return generate_moons_dataset(n_samples, noise, random_state)
    elif dataset_name == 'circles':
        return generate_circles_dataset(n_samples, noise, random_state)
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")


def normalize_data(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    return X_norm, X_min, X_max
