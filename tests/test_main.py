import pytest

from main import build_argument_parser


def test_only_run_command_is_accepted() -> None:
    parsed_arguments = build_argument_parser().parse_args(
        ["run", "--configuration", "configuration.yaml"]
    )
    assert parsed_arguments.command == "run"
    assert parsed_arguments.configuration == "configuration.yaml"


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
