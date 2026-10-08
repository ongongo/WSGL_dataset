import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score


def clustering_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate ACC with optimal label permutation, NMI, and ARI."""
    label_values = np.unique(np.concatenate((labels, predicted)))
    counts = confusion_matrix(labels, predicted, labels=label_values)
    rows, columns = linear_sum_assignment(-counts)
    return {
        "ACC": float(counts[rows, columns].sum() / labels.size),
        "NMI": float(
            normalized_mutual_info_score(
                labels, predicted, average_method="geometric"
            )
        ),
        "ARI": float(adjusted_rand_score(labels, predicted)),
    }
