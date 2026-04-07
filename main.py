from __future__ import annotations

import argparse
from pathlib import Path

from src.configuration import load_configuration
from src.pipeline import run_patch_pipeline
from src.logging import capture_runtime_logging, resolve_log_path

STAGES = [
    "select-roi",
    "preprocess",
    "segment",
    "quantify",
    "annotate",
    "spatial",
]

MODES = [
    "patch",
    "whole-slide",
]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the Orion pipeline."""
    # fmt: off
    argument_parser = argparse.ArgumentParser(
        description="Orion multiplexed imaging analysis pipeline."
    )
    argument_parser.add_argument("--configuration", required=True)
    argument_parser.add_argument("--mode", choices=MODES, default="patch")
    argument_parser.add_argument("--stage", choices=STAGES, default=None)
    # fmt: on

    return argument_parser


def main(
    argument_values: list[str] | None = None,
) -> int:
    """Parse arguments, configure logging, and dispatch to the selected pipeline mode."""
    parsed_arguments = build_argument_parser().parse_args(argument_values)

    configuration_path = parsed_arguments.configuration
    mode = parsed_arguments.mode
    stage = parsed_arguments.stage

    log_path = resolve_log_path(Path(configuration_path))
    with capture_runtime_logging(log_path) as logger:
        if mode == "whole-slide":
            logger.error(
                "Whole-slide mode is reserved but not implemented yet. Use --mode patch for now."
            )
            return 1

        run_patch_pipeline(load_configuration(configuration_path), logger, stage)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
