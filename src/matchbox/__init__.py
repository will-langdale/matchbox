from matchbox.common.logging import get_logger
from matchbox.common.exceptions import MatchboxClientSettingsException

logic_logger = get_logger("mb_logic")

try:
    # Environment variables must be loaded first for other imports to work
    from matchbox.client import *  # noqa: E402, F403
except MatchboxClientSettingsException:
    logic_logger.warning(
        "Impossible to initialise client. "
        "Please ignore if running in server mode. Otherwise, check your .env file.",
    )
