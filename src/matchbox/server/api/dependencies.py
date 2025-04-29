"""API dependencies for the Matchbox server."""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import (
    FastAPI,
    HTTPException,
    Security,
    status,
)
from fastapi.responses import Response
from fastapi.security import APIKeyHeader

from matchbox.common.logging import ASIMFormatter
from matchbox.server.api.cache import MetadataStore
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)


class ParquetResponse(Response):
    """A response object for returning parquet data."""

    media_type = "application/octet-stream"


settings: MatchboxServerSettings | None = None
backend: MatchboxDBAdapter | None = None
metadata_store = MetadataStore(expiry_minutes=30)
api_key_header = APIKeyHeader(name="X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Context manager for the FastAPI lifespan events."""
    # Set up the backend
    global settings
    global backend

    SettingsClass = get_backend_settings(MatchboxServerSettings().backend_type)
    settings = SettingsClass()
    backend = settings_to_backend(settings)

    # Define common formatter
    formatter = ASIMFormatter()

    # Configure loggers with the same handler and formatter
    loggers_to_configure = [
        "matchbox",
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "uvicorn.asgi",
        "fastapi",
    ]

    for logger_name in loggers_to_configure:
        # Configure handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(backend.settings.log_level)
        handler.setFormatter(formatter)

        logger = logging.getLogger(logger_name)
        logger.setLevel(backend.settings.log_level)
        # Remove any existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        logger.addHandler(handler)

    # Set SQLAlchemy loggers
    for sql_logger in ["sqlalchemy", "sqlalchemy.engine"]:
        logging.getLogger(sql_logger).setLevel("WARNING")

    yield

    del settings
    del backend


def validate_api_key(api_key: str = Security(api_key_header)) -> None:
    """Validate client API Key against settings."""
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing in server configuration.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required but not provided.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
    elif api_key != settings.api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key invalid.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
