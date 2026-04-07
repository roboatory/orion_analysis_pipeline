from __future__ import annotations

import logging
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import yaml


class _LoggerStream:
    def __init__(
        self,
        logger: logging.Logger,
        level: int,
    ) -> None:
        """Wrap a logger so writes to this stream are emitted as log records."""
        self._logger: logging.Logger = logger
        self._level: int = level
        self._buffer: str = ""

    def write(self, message: str) -> int:
        """Buffer incoming text and log each complete line."""
        if not message:
            return 0
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._logger.log(self._level, line.rstrip())
        return len(message)

    def flush(self) -> None:
        """Flush any remaining buffered text as a log record."""
        if self._buffer.strip():
            self._logger.log(self._level, self._buffer.rstrip())
        self._buffer = ""


def resolve_log_path(configuration_path: Path) -> Path:
    """Derive a timestamped log file path from the configuration's output directory."""
    output_directory = Path("outputs")
    if configuration_path.exists():
        with configuration_path.open("r", encoding="utf-8") as file_handle:
            configuration_payload = yaml.safe_load(file_handle) or {}
        configured_output_directory = configuration_payload.get("output_directory")
        if configured_output_directory:
            output_directory = Path(configured_output_directory)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return output_directory / "logs" / f"{timestamp}.log"


@contextmanager
def capture_runtime_logging(log_path: Path) -> Iterator[logging.Logger]:
    """Redirect stdout, stderr, and warnings to a log file for the duration of the block."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline.runtime")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.handlers = [file_handler]

    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False
    warnings_logger.handlers = [file_handler]

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_showwarning = warnings.showwarning

    logging.captureWarnings(True)
    sys.stdout = _LoggerStream(logger, logging.INFO)
    sys.stderr = _LoggerStream(logger, logging.ERROR)
    try:
        yield logger
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logging.captureWarnings(False)
        warnings.showwarning = original_showwarning
        file_handler.flush()
        file_handler.close()
        logger.handlers = []
        warnings_logger.handlers = []
