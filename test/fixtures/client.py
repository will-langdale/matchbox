from os import environ
from typing import Callable, Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from matchbox.client._settings import settings as client_settings
from matchbox.server import app


@pytest.fixture(scope="function")
def env_setter() -> Generator[Callable[[str, str], None], None, None]:
    """Set temporary env variable and refresh client settings."""
    original_values = {}

    def setter(var_name: str, var_value: str):
        original_values[var_name] = environ.get(var_name)

        environ[var_name] = var_value
        client_settings.__init__()

    yield setter

    for var_name, original_value in original_values.items():
        if original_value is None:
            del environ[var_name]
        else:
            environ[var_name] = original_value
    client_settings.__init__()


@pytest.fixture(scope="function")
def test_client() -> Generator[TestClient, None, None]:
    """Return a configured TestClient with patched backend and settings."""
    with (
        patch("matchbox.server.api.routes.settings") as mock_settings,
        patch("matchbox.server.api.routes.backend") as _,
    ):
        mock_settings.api_key = SecretStr("test-api-key")
        client = TestClient(app, headers={"X-API-Key": "test-api-key"})
        yield client
