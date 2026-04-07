from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from src.configuration import ApplicationConfiguration


@dataclass(frozen=True)
class SpatialAnalysisResult:
    cell_annotations_with_domains: pl.DataFrame
    spatial_metrics: pl.DataFrame


def compute_spatial_analysis(
    cell_annotations: pl.DataFrame,
    configuration: ApplicationConfiguration,
    random_seed: int,
) -> SpatialAnalysisResult:
    """Run neighborhood, domain, and adjacency enrichment analyses on annotated cells."""
    if cell_annotations.is_empty():
        return SpatialAnalysisResult(cell_annotations, pl.DataFrame())

    point_coordinates = np.column_stack(
        [
            cell_annotations["x_micrometers"].to_numpy(),
            cell_annotations["y_micrometers"].to_numpy(),
        ]
    )
    adjacency_edges = build_symmetric_k_nearest_neighbor_graph(
        point_coordinates,
        configuration.spatial_analysis.nearest_neighbor_count,
    )
    neighborhood_features = build_cell_neighborhood_features(
        cell_annotations,
        adjacency_edges,
    )
    domain_assignments = assign_spatial_domains(
        neighborhood_features,
        configuration.spatial_analysis.neighborhood_cluster_count,
        random_seed,
    )
    cell_annotations_with_domains = cell_annotations.join(
        domain_assignments,
        on="cell_identifier",
        how="left",
    ).with_columns(pl.col("spatial_domain").fill_null("domain_0"))
    adjacency_metrics = summarize_adjoining_cell_type_pairs(
        cell_annotations_with_domains["cell_type"].to_list(),
        adjacency_edges,
        configuration.spatial_analysis.permutation_count,
        random_seed,
    )
    spatial_domain_metrics = summarize_spatial_domains(
        neighborhood_features,
        domain_assignments,
    )
    spatial_metrics = pl.concat(
        [adjacency_metrics, spatial_domain_metrics],
        how="diagonal_relaxed",
    )
    if not spatial_metrics.is_empty():
        spatial_metrics = spatial_metrics.sort(
            ["metric_type", "group_a", "group_b"],
        )
    return SpatialAnalysisResult(
        cell_annotations_with_domains=cell_annotations_with_domains,
        spatial_metrics=spatial_metrics,
    )


def build_symmetric_k_nearest_neighbor_graph(
    point_coordinates: np.ndarray,
    nearest_neighbor_count: int,
) -> np.ndarray:
    """Build a symmetric k-NN edge list from 2D point coordinates."""
    if len(point_coordinates) == 0:
        return np.empty((0, 2), dtype=np.int32)
    kd_tree = cKDTree(point_coordinates)
    query_neighbor_count = min(
        nearest_neighbor_count + 1,
        len(point_coordinates),
    )
    _, neighbor_indices = kd_tree.query(
        point_coordinates,
        k=query_neighbor_count,
    )
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]
    edge_set = set()
    for source_index, neighbor_row in enumerate(neighbor_indices):
        for target_index in neighbor_row:
            target_index = int(target_index)
            if source_index == target_index:
                continue
            edge_set.add(tuple(sorted((source_index, target_index))))
    return np.array(sorted(edge_set), dtype=np.int32)


def build_cell_neighborhood_features(
    cell_annotations: pl.DataFrame,
    adjacency_edges: np.ndarray,
) -> pl.DataFrame:
    """Compute per-cell neighbor type fraction features from the adjacency graph."""
    cell_types = cell_annotations["cell_type"].to_list()
    unique_cell_types = sorted(set(cell_types))
    neighbor_indices_by_cell: dict[int, list[int]] = {
        cell_index: [] for cell_index in range(len(cell_types))
    }
    for source_index, target_index in adjacency_edges:
        neighbor_indices_by_cell[int(source_index)].append(int(target_index))
        neighbor_indices_by_cell[int(target_index)].append(int(source_index))
    feature_rows = []
    for cell_index, cell_type in enumerate(cell_types):
        neighbor_indices = neighbor_indices_by_cell[cell_index]
        neighbor_type_counts = Counter(
            cell_types[neighbor_index] for neighbor_index in neighbor_indices
        )
        neighbor_count = max(len(neighbor_indices), 1)
        feature_row = {
            "cell_identifier": cell_annotations[cell_index, "cell_identifier"],
            "cell_type": cell_type,
        }
        for neighbor_cell_type in unique_cell_types:
            feature_row[f"neighbor_fraction__{neighbor_cell_type}"] = (
                neighbor_type_counts.get(neighbor_cell_type, 0) / neighbor_count
            )
        feature_rows.append(feature_row)
    return pl.DataFrame(feature_rows)


def assign_spatial_domains(
    neighborhood_features: pl.DataFrame,
    neighborhood_cluster_count: int,
    random_seed: int,
) -> pl.DataFrame:
    """Cluster cells into spatial domains using k-means on neighborhood features."""
    if neighborhood_features.is_empty():
        return pl.DataFrame({"cell_identifier": [], "spatial_domain": []})
    feature_column_names = [
        column_name
        for column_name in neighborhood_features.columns
        if column_name.startswith("neighbor_fraction__")
    ]
    feature_matrix = neighborhood_features.select(feature_column_names).to_numpy()
    effective_cluster_count = min(
        neighborhood_cluster_count,
        len(neighborhood_features),
    )
    if effective_cluster_count <= 1:
        cluster_labels = np.zeros(len(neighborhood_features), dtype=int)
        stable_cluster_labels = cluster_labels
    else:
        kmeans_model = KMeans(
            n_clusters=effective_cluster_count,
            random_state=random_seed,
            n_init=10,
        )
        cluster_labels = kmeans_model.fit_predict(feature_matrix)
        stable_cluster_labels = relabel_cluster_identifiers(
            cluster_labels,
            kmeans_model.cluster_centers_,
        )
    return pl.DataFrame(
        {
            "cell_identifier": neighborhood_features["cell_identifier"],
            "spatial_domain": [
                f"domain_{cluster_label}" for cluster_label in stable_cluster_labels
            ],
        }
    )


def relabel_cluster_identifiers(
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
) -> np.ndarray:
    """Relabel cluster IDs by sorting on cluster center coordinates for deterministic ordering."""
    cluster_order = sorted(
        range(len(cluster_centers)),
        key=lambda cluster_index: tuple(cluster_centers[cluster_index].tolist()),
    )
    stable_label_by_cluster = {
        cluster_index: stable_index
        for stable_index, cluster_index in enumerate(cluster_order)
    }
    return np.array(
        [
            stable_label_by_cluster[int(cluster_label)]
            for cluster_label in cluster_labels
        ],
        dtype=int,
    )


def summarize_adjoining_cell_type_pairs(
    cell_types: list[str],
    adjacency_edges: np.ndarray,
    permutation_count: int,
    random_seed: int,
) -> pl.DataFrame:
    """Compute adjacency enrichment scores with permutation-based p-values for cell type pairs."""
    random_number_generator = np.random.default_rng(random_seed)
    observed_pair_counts = count_cell_type_pairs(
        cell_types,
        adjacency_edges,
    )
    permuted_counts_by_pair: dict[tuple[str, str], list[int]] = {
        pair: [] for pair in observed_pair_counts
    }
    label_array = np.array(cell_types, dtype=object)
    for _ in range(permutation_count):
        shuffled_labels = random_number_generator.permutation(label_array)
        shuffled_pair_counts = count_cell_type_pairs(
            list(shuffled_labels),
            adjacency_edges,
        )
        for pair in observed_pair_counts:
            permuted_counts_by_pair[pair].append(shuffled_pair_counts.get(pair, 0))
    summary_rows = []
    for pair, observed_count in observed_pair_counts.items():
        expected_count = (
            float(np.mean(permuted_counts_by_pair[pair]))
            if permuted_counts_by_pair[pair]
            else 0.0
        )
        summary_rows.append(
            {
                "metric_type": "adjacency_enrichment",
                "group_a": pair[0],
                "group_b": pair[1],
                "observed_count": observed_count,
                "expected_count": expected_count,
                "value": float(observed_count / expected_count)
                if expected_count > 0
                else 0.0,
                "empirical_p_value": compute_empirical_p_value(
                    observed_count,
                    permuted_counts_by_pair[pair],
                ),
            }
        )
    return pl.DataFrame(summary_rows).sort(["group_a", "group_b"])


def count_cell_type_pairs(
    cell_types: list[str],
    adjacency_edges: np.ndarray,
) -> Counter[tuple[str, str]]:
    """Count adjacent cell type pairs from the edge list, with pairs sorted alphabetically."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    for source_index, target_index in adjacency_edges:
        pair = tuple(
            sorted((cell_types[int(source_index)], cell_types[int(target_index)]))
        )
        pair_counts[pair] += 1
    return pair_counts


def summarize_spatial_domains(
    neighborhood_features: pl.DataFrame,
    domain_assignments: pl.DataFrame,
) -> pl.DataFrame:
    """Compute mean neighbor type fractions per spatial domain."""
    if neighborhood_features.is_empty():
        return pl.DataFrame()
    merged_features = neighborhood_features.join(
        domain_assignments,
        on="cell_identifier",
        how="left",
    )
    feature_column_names = [
        column_name
        for column_name in merged_features.columns
        if column_name.startswith("neighbor_fraction__")
    ]
    summary_rows = []
    for summary_row in (
        merged_features.group_by("spatial_domain")
        .agg(
            pl.len().alias("cell_count"),
            *[
                pl.mean(column_name).alias(column_name)
                for column_name in feature_column_names
            ],
        )
        .to_dicts()
    ):
        spatial_domain = str(summary_row["spatial_domain"])
        cell_count = int(summary_row["cell_count"])
        for column_name in feature_column_names:
            summary_rows.append(
                {
                    "metric_type": "spatial_domain_composition",
                    "group_a": spatial_domain,
                    "group_b": column_name.removeprefix("neighbor_fraction__"),
                    "observed_count": cell_count,
                    "expected_count": None,
                    "value": float(summary_row[column_name]),
                    "empirical_p_value": None,
                }
            )
    return pl.DataFrame(summary_rows).sort(["group_a", "group_b"])


def compute_empirical_p_value(
    observed_value: float,
    permuted_values: list[float],
) -> float:
    """Calculate an empirical p-value as the fraction of permutations at or above the observed value."""
    if not permuted_values:
        return 1.0
    permuted_array = np.asarray(permuted_values, dtype=float)
    return float(
        (np.sum(permuted_array >= observed_value) + 1) / (len(permuted_array) + 1)
    )
