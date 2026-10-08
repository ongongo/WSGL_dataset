"""Run the CLL_SUB_111 K-means and WSGL comparison."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.cluster import KMeans

from .dataset import load_dataset, preprocess_features
from .metrics import clustering_metrics
from .model import fit_wsgl


DATA_PATH = Path(__file__).resolve().parents[2] / "CLL_SUB_111.mat"
RESULT_PATH = Path(__file__).with_name("CLL_SUB_111_results.csv")


def run_experiment(
    dataset_path: Path,
    output_path: Path,
    runs: int,
    base_seed: int,
    subspace_count: int,
    partition_count: int,
    neighbor_count: int,
    mu: float,
    max_iterations: int,
) -> list[dict[str, float | int | str]]:
    raw_features, labels = load_dataset(dataset_path)
    features = preprocess_features(raw_features)
    class_count = np.unique(labels).size
    run_records: list[dict[str, float | int | str]] = []

    print(
        f"{dataset_path.name}: X={features.shape}, classes={class_count}, "
        f"runs={runs}"
    )
    for run_index in range(runs):
        seed = base_seed + run_index
        baseline = KMeans(
            n_clusters=class_count,
            random_state=seed,
            n_init=20,
            algorithm="lloyd",
        ).fit_predict(features)
        wsgl = fit_wsgl(
            features,
            class_count,
            seed,
            subspace_count=subspace_count,
            partition_count=partition_count,
            neighbor_count=neighbor_count,
            mu=mu,
            max_iterations=max_iterations,
        )
        for method, prediction in (("KMeans", baseline), ("WSGL", wsgl)):
            run_records.append(
                {
                    "method": method,
                    "run": run_index + 1,
                    "seed": seed,
                    **clustering_metrics(labels, prediction),
                }
            )

    summary = []
    for method in ("KMeans", "WSGL"):
        records = [record for record in run_records if record["method"] == method]
        row: dict[str, float | int | str] = {
            "dataset": "CLL_SUB_111",
            "method": method,
            "runs": len(records),
        }
        for metric in ("ACC", "NMI", "ARI"):
            values = np.asarray([record[metric] for record in records], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        summary.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print("Method  ACC (mean +/- std)  NMI (mean +/- std)  ARI (mean +/- std)")
    for row in summary:
        print(
            f"{row['method']:<7} "
            f"{row['ACC_mean']:.4f} +/- {row['ACC_std']:.4f}  "
            f"{row['NMI_mean']:.4f} +/- {row['NMI_std']:.4f}  "
            f"{row['ARI_mean']:.4f} +/- {row['ARI_std']:.4f}"
        )
    print(f"Saved: {output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subspaces", type=int, default=6)
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--mu", type=float, default=5.0)
    parser.add_argument("--max-iterations", type=int, default=50)
    args = parser.parse_args()
    run_experiment(
        args.data,
        args.output,
        args.runs,
        args.seed,
        args.subspaces,
        args.partitions,
        args.neighbors,
        args.mu,
        args.max_iterations,
    )
