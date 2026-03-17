import pytest
from pathlib import Path

from main import build_argument_parser, main

REAL_DATA_CONFIGURATION_PATH = Path(
    "/Users/rohit/Desktop/orion/configurations/CRC33_01.yaml"
)
LOG_DIRECTORY = Path("/Users/rohit/Desktop/orion/outputs/logs")


def test_only_run_command_is_accepted() -> None:
    parsed_arguments = build_argument_parser().parse_args(
        ["run", "--configuration", "configuration.yaml"]
    )
    assert parsed_arguments.command == "run"
    assert parsed_arguments.configuration == "configuration.yaml"
    assert parsed_arguments.mode == "patch"


def test_inspect_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(
            ["inspect", "--configuration", "configuration.yaml"]
        )


def test_select_patch_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(
            ["select-patch", "--configuration", "configuration.yaml"]
        )


def test_missing_run_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(["--configuration", "configuration.yaml"])


def test_missing_configuration_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(["run"])


def test_whole_slide_mode_is_accepted_by_parser() -> None:
    parsed_arguments = build_argument_parser().parse_args(
        ["run", "--configuration", "configuration.yaml", "--mode", "whole-slide"]
    )
    assert parsed_arguments.mode == "whole-slide"


def test_whole_slide_mode_returns_non_zero_until_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    existing_log_paths = set(LOG_DIRECTORY.glob("*.log"))
    assert (
        main(
            [
                "run",
                "--configuration",
                "configuration.yaml",
                "--mode",
                "whole-slide",
            ]
        )
        == 1
    )
    new_log_paths = set(LOG_DIRECTORY.glob("*.log")) - existing_log_paths
    assert len(new_log_paths) == 1
    captured_output = capsys.readouterr()
    assert captured_output.out == ""
    assert captured_output.err == ""


@pytest.mark.skipif(
    not REAL_DATA_CONFIGURATION_PATH.exists(),
    reason="Real CRC33_01 configuration is not available.",
)
def test_patch_mode_run_writes_minimal_output_set(
    capsys: pytest.CaptureFixture[str],
) -> None:
    existing_log_paths = set(LOG_DIRECTORY.glob("*.log"))
    assert (
        main(
            [
                "run",
                "--configuration",
                str(REAL_DATA_CONFIGURATION_PATH),
                "--mode",
                "patch",
            ]
        )
        == 0
    )
    output_directory = Path("/Users/rohit/Desktop/orion/outputs/CRC33_01")
    new_log_paths = set(LOG_DIRECTORY.glob("*.log")) - existing_log_paths
    assert len(new_log_paths) == 1
    log_path = next(iter(new_log_paths))
    expected_output_paths = [
        output_directory / "corrected_patch.tif",
        output_directory / "segmentation_mask.tif",
        output_directory / "segmentation_overlay.tif",
        output_directory / "cell_features.csv",
        output_directory / "cell_annotations.csv",
        output_directory / "spatial_metrics.csv",
        output_directory / "preprocessing_comparison.png",
        output_directory / "cell_type_map.png",
        output_directory / "spatial_domain_map.png",
        output_directory / "configuration_snapshot.yaml",
    ]
    for expected_output_path in expected_output_paths:
        assert expected_output_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "region_of_interest_quality:" in log_text
    assert "tissue_fraction" in log_text
    assert "informative_channel_fraction" in log_text
    captured_output = capsys.readouterr()
    assert captured_output.out == ""
    assert captured_output.err == ""
