from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans

from orion.configuration import ApplicationConfiguration


@dataclass(frozen=True)
class SpatialAnalysisOutputs:
    adjoining_cell_type_pairs: pl.DataFrame
    nearest_neighbor_distances: pl.DataFrame
    neighborhood_features: pl.DataFrame
    neighborhood_clusters: pl.DataFrame
    cell_type_clusters: pl.DataFrame


def compute_spatial_analysis_outputs(
    annotated_cell_measurements: pl.DataFrame,
    configuration: ApplicationConfiguration,
    random_seed: int = 13,
) -> SpatialAnalysisOutputs:
    if annotated_cell_measurements.is_empty():
        empty_data_frame = pl.DataFrame()
        return SpatialAnalysisOutputs(
            empty_data_frame,
            empty_data_frame,
            empty_data_frame,
            empty_data_frame,
            empty_data_frame,
        )

    point_coordinates = np.column_stack(
        [
            annotated_cell_measurements["x_micrometers"].to_numpy(),
            annotated_cell_measurements["y_micrometers"].to_numpy(),
        ]
    )
    cell_types = annotated_cell_measurements["cell_type"].to_list()
    adjacency_edges = build_symmetric_k_nearest_neighbor_graph(
        point_coordinates,
        configuration.spatial_analysis.nearest_neighbor_count,
    )
    adjoining_cell_type_pairs = summarize_adjoining_cell_type_pairs(
        cell_types,
        adjacency_edges,
        configuration.spatial_analysis.permutation_count,
        random_seed,
    )
    nearest_neighbor_distances = summarize_nearest_neighbor_distances(
        point_coordinates,
        cell_types,
        configuration,
        random_seed,
    )
    neighborhood_features = build_cell_neighborhood_features(
        annotated_cell_measurements,
        adjacency_edges,
    )
    neighborhood_clusters = cluster_cell_neighborhoods(
        neighborhood_features,
        configuration.spatial_analysis.neighborhood_cluster_count,
        random_seed,
    )
    cell_type_clusters = cluster_cell_types_spatially(
        annotated_cell_measurements,
        configuration.spatial_analysis.minimum_cells_per_type_for_clustering,
    )
    return SpatialAnalysisOutputs(
        adjoining_cell_type_pairs=adjoining_cell_type_pairs,
        nearest_neighbor_distances=nearest_neighbor_distances,
        neighborhood_features=neighborhood_features,
        neighborhood_clusters=neighborhood_clusters,
        cell_type_clusters=cell_type_clusters,
    )


def build_symmetric_k_nearest_neighbor_graph(
    point_coordinates: np.ndarray,
    nearest_neighbor_count: int,
) -> np.ndarray:
    if len(point_coordinates) == 0:
        return np.empty((0, 2), dtype=np.int32)
    kd_tree = cKDTree(point_coordinates)
    query_neighbor_count = min(nearest_neighbor_count + 1, len(point_coordinates))
    _, neighbor_indices = kd_tree.query(point_coordinates, k=query_neighbor_count)
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


def summarize_adjoining_cell_type_pairs(
    cell_types: list[str],
    adjacency_edges: np.ndarray,
    permutation_count: int,
    random_seed: int,
) -> pl.DataFrame:
    random_number_generator = np.random.default_rng(random_seed)
    observed_pair_counts = count_cell_type_pairs(cell_types, adjacency_edges)
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
                "cell_type_a": pair[0],
                "cell_type_b": pair[1],
                "observed_count": observed_count,
                "expected_count": expected_count,
                "enrichment_ratio": float(observed_count / expected_count)
                if expected_count > 0
                else 0.0,
                "empirical_p_value": compute_empirical_p_value(
                    observed_count,
                    permuted_counts_by_pair[pair],
                    tail="greater",
                ),
            }
        )
    return pl.DataFrame(summary_rows).sort(["cell_type_a", "cell_type_b"])


def count_cell_type_pairs(
    cell_types: list[str],
    adjacency_edges: np.ndarray,
) -> Counter[tuple[str, str]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    for source_index, target_index in adjacency_edges:
        pair = tuple(
            sorted((cell_types[int(source_index)], cell_types[int(target_index)]))
        )
        pair_counts[pair] += 1
    return pair_counts


def summarize_nearest_neighbor_distances(
    point_coordinates: np.ndarray,
    cell_types: list[str],
    configuration: ApplicationConfiguration,
    random_seed: int,
) -> pl.DataFrame:
    random_number_generator = np.random.default_rng(random_seed)
    unique_cell_types = sorted(set(cell_types))
    summary_rows = []
    label_array = np.array(cell_types, dtype=object)
    for source_cell_type in unique_cell_types:
        source_indices = np.where(label_array == source_cell_type)[0]
        if (
            len(source_indices)
            < configuration.spatial_analysis.minimum_cells_per_type_for_pairwise_analysis
        ):
            continue
        source_points = point_coordinates[source_indices]
        for target_cell_type in unique_cell_types:
            target_indices = np.where(label_array == target_cell_type)[0]
            if (
                len(target_indices)
                < configuration.spatial_analysis.minimum_cells_per_type_for_pairwise_analysis
            ):
                continue
            target_points = point_coordinates[target_indices]
            observed_distance = compute_median_nearest_neighbor_distance(
                source_points,
                target_points,
                same_population=source_cell_type == target_cell_type,
            )
            permuted_distances = []
            for _ in range(configuration.spatial_analysis.permutation_count):
                shuffled_indices = random_number_generator.permutation(
                    len(point_coordinates)
                )[: len(target_indices)]
                shuffled_points = point_coordinates[shuffled_indices]
                permuted_distances.append(
                    compute_median_nearest_neighbor_distance(
                        source_points,
                        shuffled_points,
                        same_population=False,
                    )
                )
            summary_rows.append(
                {
                    "source_cell_type": source_cell_type,
                    "target_cell_type": target_cell_type,
                    "observed_median_distance_micrometers": observed_distance,
                    "permuted_median_distance_micrometers": float(
                        np.mean(permuted_distances)
                    )
                    if permuted_distances
                    else 0.0,
                    "empirical_p_value": compute_empirical_p_value(
                        observed_distance,
                        permuted_distances,
                        tail="less",
                    ),
                }
            )
    return pl.DataFrame(summary_rows).sort(["source_cell_type", "target_cell_type"])


def compute_median_nearest_neighbor_distance(
    source_points: np.ndarray,
    target_points: np.ndarray,
    same_population: bool,
) -> float:
    if len(source_points) == 0 or len(target_points) == 0:
        return 0.0
    kd_tree = cKDTree(target_points)
    query_neighbor_count = 2 if same_population and len(target_points) > 1 else 1
    distances, _ = kd_tree.query(source_points, k=query_neighbor_count)
    if np.ndim(distances) == 0:
        distances = np.array([distances], dtype=float)
    if query_neighbor_count == 2:
        distances = np.asarray(distances)[:, 1]
    return float(np.median(np.asarray(distances, dtype=float)))


def build_cell_neighborhood_features(
    annotated_cell_measurements: pl.DataFrame,
    adjacency_edges: np.ndarray,
) -> pl.DataFrame:
    cell_types = annotated_cell_measurements["cell_type"].to_list()
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
        neighbor_count = len(neighbor_indices) if neighbor_indices else 1
        feature_row = {
            "cell_identifier": annotated_cell_measurements[
                cell_index, "cell_identifier"
            ],
            "cell_type": cell_type,
            "same_type_fraction": neighbor_type_counts.get(cell_type, 0)
            / neighbor_count,
            "dominant_neighbor_type": neighbor_type_counts.most_common(1)[0][0]
            if neighbor_type_counts
            else "None",
        }
        for neighbor_cell_type in unique_cell_types:
            feature_row[f"neighbor_fraction__{neighbor_cell_type}"] = (
                neighbor_type_counts.get(neighbor_cell_type, 0) / neighbor_count
            )
        feature_rows.append(feature_row)
    return pl.DataFrame(feature_rows)


def cluster_cell_neighborhoods(
    neighborhood_features: pl.DataFrame,
    neighborhood_cluster_count: int,
    random_seed: int,
) -> pl.DataFrame:
    if neighborhood_features.is_empty():
        return pl.DataFrame()
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
    else:
        kmeans_model = KMeans(
            n_clusters=effective_cluster_count,
            random_state=random_seed,
            n_init=10,
        )
        cluster_labels = kmeans_model.fit_predict(feature_matrix)
    return neighborhood_features.with_columns(
        pl.Series("neighborhood_cluster", cluster_labels)
    )


def cluster_cell_types_spatially(
    annotated_cell_measurements: pl.DataFrame,
    minimum_cells_per_type: int,
) -> pl.DataFrame:
    if annotated_cell_measurements.is_empty():
        return pl.DataFrame()
    summary_rows = []
    for cell_type in sorted(
        annotated_cell_measurements["cell_type"].unique().to_list()
    ):
        cell_type_subset = annotated_cell_measurements.filter(
            pl.col("cell_type") == cell_type
        )
        if cell_type_subset.height < minimum_cells_per_type:
            continue
        point_coordinates = np.column_stack(
            [
                cell_type_subset["x_micrometers"].to_numpy(),
                cell_type_subset["y_micrometers"].to_numpy(),
            ]
        )
        cluster_labels = DBSCAN(eps=40.0, min_samples=5).fit_predict(point_coordinates)
        clustered_mask = cluster_labels >= 0
        clustered_identifiers = cluster_labels[clustered_mask]
        cluster_sizes = Counter(clustered_identifiers)
        summary_rows.append(
            {
                "cell_type": cell_type,
                "cluster_count": len(cluster_sizes),
                "clustered_fraction": float(clustered_mask.mean())
                if len(clustered_mask)
                else 0.0,
                "median_cluster_size": float(np.median(list(cluster_sizes.values())))
                if cluster_sizes
                else 0.0,
            }
        )
    return pl.DataFrame(summary_rows)


def compute_empirical_p_value(
    observed_value: float,
    permuted_values: list[float],
    tail: str,
) -> float:
    if not permuted_values:
        return 1.0
    permuted_array = np.asarray(permuted_values, dtype=float)
    if tail == "greater":
        extreme_count = np.count_nonzero(permuted_array >= observed_value)
    else:
        extreme_count = np.count_nonzero(permuted_array <= observed_value)
    return float((extreme_count + 1) / (len(permuted_array) + 1))
