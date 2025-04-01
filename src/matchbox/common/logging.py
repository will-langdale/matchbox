"""Logging utilities."""

import logging
from typing import Any, Final

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class PrefixedLoggerAdapter(logging.LoggerAdapter):
    """A logger adapter that supports adding a prefix enclosed in square brackets.

    This adapter allows passing an optional prefix parameter to any logging call
    without modifying the underlying logger.
    """

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Process the log message, adding a prefix if provided.

        Args:
            msg: The log message
            kwargs: Additional arguments to the logging method

        Returns:
            Tuple of (modified_message, modified_kwargs)
        """
        prefix = kwargs.pop("prefix", None)

        if prefix:
            msg = f"[{prefix}] {msg}"

        return msg, kwargs


logger: Final[logging.Logger] = logging.getLogger("matchbox")
"""Logger for Matchbox.

Used for all logging in the Matchbox library.
"""
logger.addHandler(logging.NullHandler())
logger = PrefixedLoggerAdapter(logger, {})


console: Final[Console] = Console()
"""Console for Matchbox.

Used for any CLI utilities in the Matchbox library.
"""


def build_progress_bar(console_: Console | None = None) -> Progress:
    """Create a progress bar."""
    if console_ is None:
        console_ = console

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console_,
    )
