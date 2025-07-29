"""Logging utilities."""

import json
import logging
import os
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Final, Literal

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
"""Type for all Python log levels."""


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


logger: Final[PrefixedLoggerAdapter] = PrefixedLoggerAdapter(
    logging.getLogger("matchbox"), {}
)
"""Logger for Matchbox.

Used for all logging in the Matchbox library.

Allows passing a prefix to any logging call.

Examples:
    ```python
    log_prefix = f"Model {name}"
    logger.debug("Inserting metadata", prefix=log_prefix)
    logger.debug("Inserting data", prefix=log_prefix)
    logger.info("Insert successful", prefix=log_prefix)
    ```
"""

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


class ASIMFormatter(logging.Formatter):
    """Format logging with ASIM standard fields."""

    _tracer = None
    """Datadog tracer instance."""

    def _get_first_64_bits_of(self, trace_id):
        return str((1 << 64) - 1 & trace_id)

    @cached_property
    def event_severity(self) -> dict[str, str]:
        """Event severity level lookup."""
        return {
            "DEBUG": "Informational",
            "INFO": "Informational",
            "WARNING": "Low",
            "ERROR": "Medium",
            "CRITICAL": "High",
        }

    @cached_property
    def container_id(self) -> str:
        """AWS ECS container ID.

        This environment variable is injected by AWS to all ECS tasks.
        """
        uri = os.environ.get("ECS_CONTAINER_METADATA_URI", "")
        return uri.split("/")[-1] if uri else ""

    @cached_property
    def env(self) -> str:
        """Datadog's environment value.

        This environment variable is set by the ECS task definition.
        """
        return os.getenv("DD_ENV", default="")

    @cached_property
    def service(self) -> str:
        """Datadog's service value.

        This environment variable is set by the ECS task definition.
        """
        return os.getenv("DD_SERVICE", default="")

    @cached_property
    def version(self) -> str:
        """Datadog's version value.

        This environment variable is set in the Dockerfile for the task.
        """
        return os.getenv("DD_VERSION", default="")

    def get_trace_id_span_id(self) -> tuple[str | None, str | None]:
        """Retrieve's Datadog's trace ID and span ID variables.

        These two variables are discovered by the Datadog Python tracing library.
        """
        # ddtrace is a server-side dependency
        # imported here as logging.py is in common, and is imported
        # client side as well
        if self._tracer is None:
            from ddtrace.trace import tracer

            self._tracer = tracer

        span = self._tracer.current_span()
        trace_id, span_id = (
            (self._get_first_64_bits_of(span.trace_id), span.span_id)
            if span
            else (None, None)
        )
        return trace_id, span_id

    def format(self, record) -> str:
        """Convert logs to JSON."""
        log_time = datetime.fromtimestamp(record.created, timezone.utc).isoformat()
        trace_id, span_id = self.get_trace_id_span_id()
        return json.dumps(
            {
                "EventCount": 1,
                "EventStartTime": log_time,
                "EventEndTime": log_time,
                "EventType": record.name,
                "EventSeverity": self.event_severity[record.levelname],
                "EventOriginalSeverity": record.levelname,
                "dd.application": "matchbox",
                "dd.container_id": self.container_id,
                "dd.env": self.env,
                "dd.service": self.service,
                "dd.source": "python",
                "dd.sourcecategory": "sourcecode",
                "dd.span_id": span_id,
                "dd.team": "matchbox",
                "dd.trace_id": trace_id,
                "dd.version": self.version,
                "message": record.getMessage(),
            },
        )
