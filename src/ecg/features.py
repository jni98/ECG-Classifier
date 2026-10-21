import numpy as np


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extract features per ecg channel

    Args:
        X (np.ndarray): The input ECG signals

    Returns:
        np.ndarray: The extracted features
    """
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    max_val = np.max(X, axis=1)
    min_val = np.min(X, axis=1)

    return np.concatenate([mean, std, max_val, min_val], axis=1)
