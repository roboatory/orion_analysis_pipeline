import numpy as np

from src.data_models import RegionOfInterestBox
from src.quantification import quantify_cells_in_region_of_interest


def test_quantification_schema_and_values() -> None:
    label_image = np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
            [2, 2, 0],
        ],
        dtype=np.int32,
    )
    intensity_image_by_marker = {
        "Hoechst": np.ones((3, 3), dtype=float),
        "CD45": np.full((3, 3), 2.0, dtype=float),
    }
    quantified_data_frame = quantify_cells_in_region_of_interest(
        label_image,
        intensity_image_by_marker,
        ["Hoechst", "CD45"],
        RegionOfInterestBox(10, 20, 3, 3),
        0.5,
        0.5,
    )
    assert {
        "cell_identifier",
        "x_micrometers",
        "y_micrometers",
        "area_square_micrometers",
        "Hoechst",
        "CD45",
    } <= set(quantified_data_frame.columns)
    assert quantified_data_frame.height == 2
    assert quantified_data_frame["area_square_micrometers"].to_list() == [1.0, 0.5]
