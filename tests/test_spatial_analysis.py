import polars as pl

from src.spatial_analysis import (
    build_radius_neighbor_graph,
    compute_spatial_analysis,
)


class DummySpatialAnalysisConfiguration:
    neighborhood_radius_micrometers = 5.0
    minimum_cells_per_type_for_pairwise_analysis = 0
    permutation_count = 5
    neighborhood_cluster_count = 2


class DummyApplicationConfiguration:
    spatial_analysis = DummySpatialAnalysisConfiguration()


def test_radius_neighbor_graph_is_symmetric() -> None:
    import numpy as np

    point_coordinates = np.array([[0, 0], [1, 0], [5, 0], [6, 0]], dtype=float)
    adjacency_edges = build_radius_neighbor_graph(point_coordinates, 1.5)
    edge_set = {tuple(edge) for edge in adjacency_edges.tolist()}
    assert (0, 1) in edge_set
    assert (2, 3) in edge_set
    assert (1, 2) not in edge_set


def test_spatial_analysis_output_is_reproducible() -> None:
    cell_annotations = pl.DataFrame(
        {
            "cell_identifier": [1, 2, 3, 4],
            "x_micrometers": [0.0, 1.0, 10.0, 11.0],
            "y_micrometers": [0.0, 0.0, 0.0, 0.0],
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
            "x_micrometers": [0.0, 2.0, 20.0, 22.0],
            "y_micrometers": [0.0, 0.0, 0.0, 0.0],
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


def test_isolated_cells_get_isolated_domain() -> None:
    cell_annotations = pl.DataFrame(
        {
            "cell_identifier": [1, 2, 3],
            "x_micrometers": [0.0, 1.0, 1000.0],
            "y_micrometers": [0.0, 0.0, 0.0],
            "cell_type": ["A", "A", "B"],
        }
    )
    result = compute_spatial_analysis(
        cell_annotations,
        DummyApplicationConfiguration(),
        random_seed=1,
    )
    domains = result.cell_annotations_with_domains
    isolated_row = domains.filter(pl.col("cell_identifier") == 3)
    assert isolated_row[0, "spatial_domain"] == "isolated"


def test_depleted_pair_is_reported() -> None:
    cell_annotations = pl.DataFrame(
        {
            "cell_identifier": [1, 2, 3, 4],
            "x_micrometers": [0.0, 1.0, 3.0, 4.0],
            "y_micrometers": [0.0, 0.0, 0.0, 0.0],
            "cell_type": ["A", "A", "B", "B"],
        }
    )
    result = compute_spatial_analysis(
        cell_annotations,
        DummyApplicationConfiguration(),
        random_seed=1,
    )
    adjacency_rows = result.spatial_metrics.filter(
        pl.col("metric_type") == "adjacency_enrichment"
    )
    pairs = {(row["group_a"], row["group_b"]) for row in adjacency_rows.to_dicts()}
    assert ("A", "B") in pairs


def test_minimum_cells_per_type_filters_rare_types() -> None:
    import numpy as np

    from src.spatial_analysis import summarize_adjoining_cell_type_pairs

    cell_types = ["A"] * 10 + ["B"] * 2
    edges = np.array([[0, 1], [2, 3], [10, 11]], dtype=np.int32)
    result = summarize_adjoining_cell_type_pairs(
        cell_types,
        edges,
        permutation_count=5,
        random_seed=1,
        minimum_cells_per_type=5,
    )
    pairs = {(row["group_a"], row["group_b"]) for row in result.to_dicts()}
    assert ("A", "A") in pairs
    assert ("A", "B") not in pairs
    assert ("B", "B") not in pairs
