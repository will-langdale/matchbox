from matchbox.common.exceptions import MatchboxClientSettingsException
from matchbox.common.logging import mb_server_logger

try:
    # Environment variables must be loaded first for other imports to work
    from matchbox.client import *  # noqa: E402, F403
except MatchboxClientSettingsException:
    mb_server_logger.warning(
        "Impossible to initialise client. "
        "Please ignore if running in server mode. Otherwise, check your .env file.",
    )
