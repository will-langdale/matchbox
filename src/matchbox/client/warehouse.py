"""Utilities to interact with a data warehouse."""

from sqlalchemy import Engine, create_engine

from matchbox.client._settings import settings
from matchbox.common.logging import logger


def _engine_fallback(engine: Engine | None = None):
    """Returns passed engine or looks for a default one."""
    if not engine:
        if default_engine := settings.default_warehouse:
            engine = create_engine(default_engine)
            logger.warning("Using default engine")
        else:
            raise ValueError(
                "An engine needs to be provided if "
                "`MB__CLIENT__DEFAULT_WAREHOUSE` is unset"
            )
    return engine
