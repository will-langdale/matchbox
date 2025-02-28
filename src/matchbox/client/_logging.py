"""Client-side logging utilities."""

from matchbox.common.logging import INFO, get_logger

client_logger = get_logger(__name__, "%(levelname)s: %(message)s")
client_logger.setLevel(INFO)
