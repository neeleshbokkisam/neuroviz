"""
Dataset loaders for synthetic 2D classification datasets.
"""

import numpy as np
from sklearn.datasets import make_moons, make_circles


def generate_xor_dataset(n_samples=200, noise=0.1, random_state=42):
    """
    Generate XOR dataset.
    
    Args:
        n_samples: Number of samples
        noise: Amount of noise to add
        random_state: Random seed
    
    Returns:
        X, y: Features and labels
    """
    np.random.seed(random_state)
    
    # Generate XOR pattern
    n = n_samples // 4
    X1 = np.random.randn(n, 2) * noise + np.array([0, 0])
    X2 = np.random.randn(n, 2) * noise + np.array([1, 1])
    X3 = np.random.randn(n, 2) * noise + np.array([0, 1])
    X4 = np.random.randn(n, 2) * noise + np.array([1, 0])
    
    X = np.vstack([X1, X2, X3, X4])
    y = np.hstack([np.zeros(n), np.zeros(n), np.ones(n), np.ones(n)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y.reshape(-1, 1)


def generate_moons_dataset(n_samples=200, noise=0.1, random_state=42):
    """
    Generate two moons dataset.
    
    Args:
        n_samples: Number of samples
        noise: Amount of noise
        random_state: Random seed
    
    Returns:
        X, y: Features and labels
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y.reshape(-1, 1)


def generate_circles_dataset(n_samples=200, noise=0.1, random_state=42):
    """
    Generate concentric circles dataset.
    
    Args:
        n_samples: Number of samples
        noise: Amount of noise
        random_state: Random seed
    
    Returns:
        X, y: Features and labels
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state, factor=0.5)
    return X, y.reshape(-1, 1)


def load_dataset(dataset_name, n_samples=200, noise=0.1, random_state=42):
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of dataset ('xor', 'moons', 'circles')
        n_samples: Number of samples
        noise: Amount of noise
        random_state: Random seed
    
    Returns:
        X, y: Features and labels
    """
    if dataset_name == 'xor':
        return generate_xor_dataset(n_samples, noise, random_state)
    elif dataset_name == 'moons':
        return generate_moons_dataset(n_samples, noise, random_state)
    elif dataset_name == 'circles':
        return generate_circles_dataset(n_samples, noise, random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def normalize_data(X):
    """
    Normalize data to [0, 1] range for better training.
    
    Args:
        X: Input features
    
    Returns:
        Normalized features and normalization parameters
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    return X_norm, X_min, X_max


