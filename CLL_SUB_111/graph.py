import numpy as np
from scipy.spatial.distance import cdist


def project_simplex(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)[::-1]
    shifted_cumsum = np.cumsum(ordered) - 1.0
    indices = np.arange(1, values.size + 1)
    active = ordered - shifted_cumsum / indices > 0
    last = np.flatnonzero(active)[-1]
    threshold = shifted_cumsum[last] / (last + 1)
    return np.maximum(values - threshold, 0.0)


def update_weights(
    distances: np.ndarray, graph: np.ndarray, gamma: float
) -> np.ndarray:
    costs = np.einsum("bij,ij->b", distances, graph)
    base = np.maximum(-costs / gamma, np.finfo(float).eps)
    return base ** (1.0 / (gamma - 1.0))


def update_graph(
    distances: np.ndarray,
    embedding: np.ndarray,
    weights: np.ndarray,
    mu: float,
    neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = embedding.shape[0]
    combined = np.tensordot(weights, distances, axes=(0, 0))
    combined += mu * cdist(embedding, embedding, metric="sqeuclidean")
    np.fill_diagonal(combined, np.inf)

    k = min(neighbor_count, count - 1)
    graph = np.zeros((count, count), dtype=np.float64)
    row_beta = np.empty(count, dtype=np.float64)
    for row_index, row in enumerate(combined):
        neighbors = np.argpartition(row, k - 1)[:k]
        neighbors = neighbors[np.argsort(row[neighbors])]
        neighbor_distances = row[neighbors]
        if k < count - 1:
            next_distance = np.partition(row, k)[k]
            beta = (k * next_distance - neighbor_distances.sum()) / 2.0
        else:
            beta = float(np.std(neighbor_distances))
        scale = max(1.0, float(np.max(np.abs(neighbor_distances))))
        beta = max(beta, np.finfo(float).eps * scale)
        row_beta[row_index] = beta
        graph[row_index, neighbors] = project_simplex(
            -neighbor_distances / (2.0 * beta)
        )
    return graph, row_beta


def graph_laplacian(graph: np.ndarray) -> np.ndarray:
    symmetric = (graph + graph.T) / 2.0
    np.fill_diagonal(symmetric, 0.0)
    return np.diag(symmetric.sum(axis=1)) - symmetric
