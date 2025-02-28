"""Client-side logging utilities."""

from matchbox.common.logging import get_logger, INFO

client_logger = get_logger(__name__, "%(levelname)s: %(message)s")
client_logger.setLevel(INFO)


