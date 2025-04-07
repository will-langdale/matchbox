"""Matchbox server.

Includes the API, and database adapters for various backends.
"""

from matchbox.server.api.routes import app
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxServerSettings,
    initialise_matchbox,
)

__all__ = ["app", "MatchboxDBAdapter", "MatchboxServerSettings"]

initialise_matchbox()
