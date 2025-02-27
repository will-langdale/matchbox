import logging

from dotenv import find_dotenv, load_dotenv

from matchbox.common.exceptions import MatchboxClientSettingsException

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

logic_logger = logging.getLogger("mb_logic")

try:
    # Environment variables must be loaded first for other imports to work
    from matchbox.client import *  # noqa: E402, F403
except MatchboxClientSettingsException:
    logic_logger.warning(
        "Impossible to initialise client. "
        "Please ignore if running in server-mode. Otherwise, check your env file.",
    )
