"""Matchbox."""

from importlib.metadata import PackageNotFoundError, version

from matchbox.common.exceptions import MatchboxClientSettingsException
from matchbox.common.logging import logger

try:
    # Environment variables must be loaded first for other imports to work
    from matchbox.client import *  # noqa: E402, F403
except MatchboxClientSettingsException:
    logger.warning(
        "Impossible to initialise client. "
        "Please ignore if running in server mode. Otherwise, check your .env file.",
    )

try:
    __version__ = version("matchbox-db")
except PackageNotFoundError:
    # package is not installed
    pass
