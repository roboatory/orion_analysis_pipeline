import pytest
from pathlib import Path

from main import build_argument_parser, main
from src.constants import STAGES

LOG_DIRECTORY = Path("/Users/rohit/Desktop/orion/outputs/logs")


def test_required_arguments_are_parsed() -> None:
    parsed_arguments = build_argument_parser().parse_args(
        ["--configuration", "configuration.yaml", "--mode", "patch"]
    )
    assert parsed_arguments.configuration == "configuration.yaml"
    assert parsed_arguments.mode == "patch"
    assert parsed_arguments.stage is None


def test_stage_flag_is_accepted_by_parser() -> None:
    parsed_arguments = build_argument_parser().parse_args(
        [
            "--configuration",
            "configuration.yaml",
            "--mode",
            "patch",
            "--stage",
            "segment",
        ]
    )
    assert parsed_arguments.stage == "segment"


def test_invalid_stage_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(
            [
                "--configuration",
                "configuration.yaml",
                "--mode",
                "patch",
                "--stage",
                "invalid",
            ]
        )


def test_all_stage_names_are_accepted_by_parser() -> None:
    for stage_name in STAGES:
        parsed_arguments = build_argument_parser().parse_args(
            [
                "--configuration",
                "configuration.yaml",
                "--mode",
                "patch",
                "--stage",
                stage_name,
            ]
        )
        assert parsed_arguments.stage == stage_name


def test_missing_configuration_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(["--mode", "patch"])


def test_missing_mode_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(["--configuration", "configuration.yaml"])


def test_whole_slide_mode_is_accepted_by_parser() -> None:
    parsed_arguments = build_argument_parser().parse_args(
        ["--configuration", "configuration.yaml", "--mode", "whole-slide"]
    )
    assert parsed_arguments.mode == "whole-slide"


def test_whole_slide_mode_returns_non_zero_until_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    existing_log_paths = set(LOG_DIRECTORY.glob("*.log"))
    assert (
        main(
            [
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
