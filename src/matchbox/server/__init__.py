from matchbox.server.api import app
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxSettings,
    initialise_matchbox,
    inject_backend,
)

__all__ = ["app", "MatchboxDBAdapter", "MatchboxSettings", "inject_backend"]

initialise_matchbox()
