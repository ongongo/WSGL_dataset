from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the MAT file's X and Y arrays."""
    data = loadmat(path)
    if "X" not in data or "Y" not in data:
        raise ValueError("MAT file must contain variables X and Y.")

    features = np.asarray(data["X"], dtype=np.float64)
    labels = np.asarray(data["Y"]).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("X must have one row for each label in Y.")
    if not np.isfinite(features).all():
        raise ValueError("X contains NaN or infinite values.")
    return features, labels


def preprocess_features(features: np.ndarray) -> np.ndarray:
    """Apply the shared column-wise scaling used by both clustering methods."""
    return MinMaxScaler().fit_transform(features)
