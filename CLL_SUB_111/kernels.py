import numpy as np
from scipy.spatial.distance import cdist


KERNELS = ("linear", "rbf", "polynomial")


def center_kernel(kernel: np.ndarray) -> np.ndarray:
    return (
        kernel
        - kernel.mean(axis=1, keepdims=True)
        - kernel.mean(axis=0, keepdims=True)
        + kernel.mean()
    )


def kernel_pca(block: np.ndarray, kernel_name: str) -> np.ndarray:
    sample_count, feature_count = block.shape
    dot = block @ block.T
    if kernel_name == "linear":
        kernel = dot
    elif kernel_name == "rbf":
        sigma = 1.0 / feature_count
        distances = cdist(block, block, metric="sqeuclidean")
        kernel = np.exp(-distances / (2.0 * sigma**2))
    elif kernel_name == "polynomial":
        kernel = (dot + 1.0) ** 3
    else:
        raise ValueError(f"Unsupported kernel: {kernel_name}")

    centered = center_kernel(kernel)
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1]
    component_count = min(int(np.ceil(0.5 * feature_count)), sample_count - 1)
    selected = order[:component_count]
    return centered @ eigenvectors[:, selected]


def make_subspaces(
    features: np.ndarray,
    subspace_count: int,
    partition_count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    subspaces = []
    for _ in range(subspace_count):
        partitions = np.array_split(
            rng.permutation(features.shape[1]), partition_count
        )
        kernels = rng.permutation(KERNELS)
        blocks = [
            kernel_pca(features[:, indices], kernels[index % len(kernels)])
            for index, indices in enumerate(partitions)
        ]
        subspaces.append(np.concatenate(blocks, axis=1))
    return subspaces
