import polars as pl

from src.spatial_analysis import (
    build_symmetric_k_nearest_neighbor_graph,
    compute_spatial_analysis,
)


class DummySpatialAnalysisConfiguration:
    nearest_neighbor_count = 2
    permutation_count = 5
    neighborhood_cluster_count = 2


class DummyApplicationConfiguration:
    spatial_analysis = DummySpatialAnalysisConfiguration()


def test_k_nearest_neighbor_graph_is_symmetric() -> None:
    import numpy as np

    point_coordinates = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    adjacency_edges = build_symmetric_k_nearest_neighbor_graph(point_coordinates, 1)
    edge_set = {tuple(edge) for edge in adjacency_edges.tolist()}
    assert (0, 1) in edge_set
    assert (2, 3) in edge_set


def test_spatial_analysis_output_is_reproducible() -> None:
    cell_annotations = pl.DataFrame(
        {
            "cell_identifier": [1, 2, 3, 4],
            "x_micrometers": [0.0, 1.0, 10.0, 11.0],
            "y_micrometers": [0.0, 0.0, 0.0, 0.0],
            "x_pixels": [0.0, 1.0, 10.0, 11.0],
            "y_pixels": [0.0, 0.0, 0.0, 0.0],
            "cell_type": ["A", "A", "B", "B"],
        }
    )
    first_output = compute_spatial_analysis(
        cell_annotations,
        DummyApplicationConfiguration(),
        random_seed=3,
    )
    second_output = compute_spatial_analysis(
        cell_annotations,
        DummyApplicationConfiguration(),
        random_seed=3,
    )
    assert (
        first_output.spatial_metrics.to_dicts()
        == second_output.spatial_metrics.to_dicts()
    )


def test_spatial_domains_are_added_to_annotations() -> None:
    cell_annotations = pl.DataFrame(
        {
            "cell_identifier": [1, 2, 3, 4],
            "x_micrometers": [0.0, 1.0, 10.0, 11.0],
            "y_micrometers": [0.0, 0.0, 0.0, 0.0],
            "x_pixels": [0.0, 2.0, 20.0, 22.0],
            "y_pixels": [0.0, 0.0, 0.0, 0.0],
            "cell_type": ["A", "A", "B", "B"],
        }
    )
    spatial_analysis_result = compute_spatial_analysis(
        cell_annotations,
        DummyApplicationConfiguration(),
        random_seed=3,
    )
    assert (
        "spatial_domain"
        in spatial_analysis_result.cell_annotations_with_domains.columns
    )
    assert "metric_type" in spatial_analysis_result.spatial_metrics.columns
