import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from .graph import graph_laplacian, update_graph, update_weights
from .kernels import make_subspaces


def fit_wsgl(
    features: np.ndarray,
    cluster_count: int,
    seed: int,
    subspace_count: int = 6,
    partition_count: int = 10,
    neighbor_count: int = 10,
    mu: float = 5.0,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Alternately update subspace weights, adaptive graph, and embedding."""
    rng = np.random.default_rng(seed)
    subspaces = make_subspaces(features, subspace_count, partition_count, rng)
    distances = np.stack(
        [cdist(space, space, metric="sqeuclidean") for space in subspaces]
    )
    count = features.shape[0]
    embedding = np.linalg.qr(
        rng.standard_normal((count, cluster_count)), mode="reduced"
    )[0]
    graph = np.zeros((count, count), dtype=np.float64)
    for row_index in range(count):
        neighbors = np.delete(np.arange(count), row_index)
        graph[row_index, neighbors] = rng.dirichlet(np.ones(count - 1))

    previous_objective: float | None = None
    weights = np.full(subspace_count, 1.0 / subspace_count)
    row_beta = np.ones(count)
    for _ in range(max_iterations):
        weights = update_weights(distances, graph, gamma=-1.0)
        graph, row_beta = update_graph(
            distances, embedding, weights, mu, neighbor_count
        )
        laplacian = graph_laplacian(graph)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        embedding = eigenvectors[:, np.argsort(eigenvalues)[:cluster_count]]

        graph_cost = float(np.einsum("b,bij,ij->", weights, distances, graph))
        regularization = float(np.dot(row_beta, np.square(graph).sum(axis=1)))
        spectral_cost = float(mu * np.trace(embedding.T @ laplacian @ embedding))
        weight_cost = float(np.sum(weights**-1.0))
        objective = graph_cost + regularization + spectral_cost + weight_cost
        if previous_objective is not None and abs(previous_objective - objective) <= (
            tolerance * max(1.0, abs(previous_objective))
        ):
            break
        previous_objective = objective

    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding /= np.maximum(row_norms, np.finfo(float).eps)
    return KMeans(
        n_clusters=cluster_count,
        random_state=seed,
        n_init=20,
        algorithm="lloyd",
    ).fit_predict(embedding)
