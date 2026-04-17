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
    adjacency_edges = build_radius_neighbor_graph(
        point_coordinates,
        configuration.spatial_analysis.neighborhood_radius_micrometers,
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
    ).with_columns(pl.col("spatial_domain").fill_null("isolated"))
    adjacency_metrics = summarize_adjoining_cell_type_pairs(
        cell_annotations_with_domains["cell_type"].to_list(),
        adjacency_edges,
        configuration.spatial_analysis.permutation_count,
        random_seed,
        configuration.spatial_analysis.minimum_cells_per_type_for_pairwise_analysis,
    )
    spatial_metrics = adjacency_metrics
    if not spatial_metrics.is_empty():
        spatial_metrics = spatial_metrics.sort(
            ["metric_type", "group_a", "group_b"],
        )
    return SpatialAnalysisResult(
        cell_annotations_with_domains=cell_annotations_with_domains,
        spatial_metrics=spatial_metrics,
    )


def build_radius_neighbor_graph(
    point_coordinates: np.ndarray,
    radius_micrometers: float,
) -> np.ndarray:
    """Build a symmetric edge list from all point pairs within a radius."""
    if len(point_coordinates) == 0:
        return np.empty((0, 2), dtype=np.int32)
    kd_tree = cKDTree(point_coordinates)
    neighbor_lists = kd_tree.query_ball_point(point_coordinates, r=radius_micrometers)
    edge_set: set[tuple[int, int]] = set()
    for source_index, neighbor_indices in enumerate(neighbor_lists):
        for target_index in neighbor_indices:
            if source_index == target_index:
                continue
            edge_set.add(tuple(sorted((source_index, target_index))))
    if not edge_set:
        return np.empty((0, 2), dtype=np.int32)
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
        is_isolated = len(neighbor_indices) == 0
        neighbor_type_counts = Counter(
            cell_types[neighbor_index] for neighbor_index in neighbor_indices
        )
        neighbor_count = max(len(neighbor_indices), 1)
        feature_row = {
            "cell_identifier": cell_annotations[cell_index, "cell_identifier"],
            "cell_type": cell_type,
            "is_isolated": is_isolated,
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
    connected_cells = neighborhood_features.filter(~pl.col("is_isolated"))
    isolated_cells = neighborhood_features.filter(pl.col("is_isolated"))
    isolated_assignments = pl.DataFrame(
        {
            "cell_identifier": isolated_cells["cell_identifier"],
            "spatial_domain": ["isolated"] * len(isolated_cells),
        }
    )
    if connected_cells.is_empty():
        return isolated_assignments
    feature_column_names = [
        column_name
        for column_name in connected_cells.columns
        if column_name.startswith("neighbor_fraction__")
    ]
    feature_matrix = connected_cells.select(feature_column_names).to_numpy()
    effective_cluster_count = min(
        neighborhood_cluster_count,
        len(connected_cells),
    )
    if effective_cluster_count <= 1:
        cluster_labels = np.zeros(len(connected_cells), dtype=int)
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
    connected_assignments = pl.DataFrame(
        {
            "cell_identifier": connected_cells["cell_identifier"],
            "spatial_domain": [
                f"domain_{cluster_label}" for cluster_label in stable_cluster_labels
            ],
        }
    )
    return pl.concat([connected_assignments, isolated_assignments], how="vertical")


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
    minimum_cells_per_type: int = 0,
) -> pl.DataFrame:
    """Compute adjacency enrichment z-scores with permutation-based p-values for all cell type pairs."""
    type_counts = Counter(cell_types)
    eligible_types = sorted(
        cell_type
        for cell_type, count in type_counts.items()
        if count >= minimum_cells_per_type
    )
    all_pairs = [
        (eligible_types[i], eligible_types[j])
        for i in range(len(eligible_types))
        for j in range(i, len(eligible_types))
    ]
    if not all_pairs:
        return pl.DataFrame()
    random_number_generator = np.random.default_rng(random_seed)
    observed_pair_counts = count_cell_type_pairs(cell_types, adjacency_edges)
    permuted_counts_by_pair: dict[tuple[str, str], list[int]] = {
        pair: [] for pair in all_pairs
    }
    label_array = np.array(cell_types, dtype=object)
    for _ in range(permutation_count):
        shuffled_labels = random_number_generator.permutation(label_array)
        shuffled_pair_counts = count_cell_type_pairs(
            list(shuffled_labels),
            adjacency_edges,
        )
        for pair in all_pairs:
            permuted_counts_by_pair[pair].append(shuffled_pair_counts.get(pair, 0))
    summary_rows = []
    for pair in all_pairs:
        observed_count = observed_pair_counts.get(pair, 0)
        permuted_values = permuted_counts_by_pair[pair]
        expected_count = float(np.mean(permuted_values)) if permuted_values else 0.0
        permuted_std = float(np.std(permuted_values)) if permuted_values else 0.0
        z_score = (
            float((observed_count - expected_count) / permuted_std)
            if permuted_std > 0
            else 0.0
        )
        summary_rows.append(
            {
                "metric_type": "adjacency_enrichment",
                "group_a": pair[0],
                "group_b": pair[1],
                "observed_count": observed_count,
                "expected_count": expected_count,
                "z_score": z_score,
                "empirical_p_value": compute_empirical_p_value(
                    observed_count,
                    permuted_values,
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
