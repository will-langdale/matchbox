from os import environ
from typing import Callable, Generator
from unittest.mock import patch

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Client
from pydantic import SecretStr
from respx import MockRouter

from matchbox.client._handler import create_client
from matchbox.client._settings import ClientSettings
from matchbox.client._settings import settings as client_settings
from matchbox.client.authorisation import (
    generate_EdDSA_key_pair,
    generate_json_web_token,
)
from matchbox.server.api import app
from matchbox.server.api.dependencies import backend
from matchbox.server.api.dependencies import settings as settings_dependency


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
def test_client(env_setter) -> Generator[TestClient, None, None]:
    """Return a configured TestClient with patched backend and settings."""
    with (
        patch("matchbox.server.api.dependencies.settings") as mock_settings,
        patch("matchbox.server.api.dependencies.backend") as mock_backend,
    ):
        # Generate private and public key pair
        private_key, public_key = generate_EdDSA_key_pair()

        mock_settings.authorisation = True
        mock_settings.public_key = SecretStr(public_key.decode())
        env_setter("MB__CLIENT__PRIVATE_KEY", private_key.decode())

        app.dependency_overrides[backend] = lambda: mock_backend
        app.dependency_overrides[settings_dependency] = lambda: mock_settings

        token = generate_json_web_token(sub="test.user@email.com")
        yield TestClient(app, headers={"Authorization": token})

        app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def matchbox_api() -> Generator[MockRouter, None, None]:
    """Client for the mocked Matchbox API."""
    with respx.mock(
        base_url=client_settings.api_root, assert_all_called=True
    ) as respx_mock:
        yield respx_mock


@pytest.fixture(scope="session")
def matchbox_client_settings() -> ClientSettings:
    """Client settings for the Matchbox API running in Docker."""
    return client_settings


@pytest.fixture(scope="session")
def matchbox_client(matchbox_client_settings: ClientSettings) -> Client:
    """Client for the Matchbox API running in Docker."""
    return create_client(settings=matchbox_client_settings)
