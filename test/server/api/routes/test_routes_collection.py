"""Unit tests for collection and version management endpoints."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from matchbox.common.dtos import (
    BackendResourceType,
    Collection,
    OKMessage,
    Version,
)
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxCollectionNotFoundError,
    MatchboxDeletionNotConfirmed,
    MatchboxVersionAlreadyExists,
)
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import source_factory

# Collection management tests


def test_list_collections(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test listing all collections."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.list_collections = Mock(return_value=["default", "test_collection"])

    response = test_client.get("/collections")

    assert response.status_code == 200
    assert response.json() == ["default", "test_collection"]
    mock_backend.list_collections.assert_called_once()


def test_get_collection(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test retrieving a specific collection."""
    test_client, mock_backend, _ = api_client_and_mocks

    # Mock collection data with versions and resolutions
    source_resolution = source_factory().source.to_resolution()
    model_resolution = model_factory().model.to_resolution()
    collection_data = {
        "v1": [source_resolution, model_resolution],
        "v2": [source_resolution],
    }

    mock_backend.get_collection = Mock(return_value=collection_data)

    response = test_client.get("/collections/test_collection")

    assert response.status_code == 200
    assert "v1" in response.json()
    assert "v2" in response.json()
    assert len(response.json()["v1"]) == 2
    assert len(response.json()["v2"]) == 1
    mock_backend.get_collection.assert_called_once_with(name="test_collection")


def test_get_collection_404(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test retrieving a non-existent collection."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_collection = Mock(side_effect=MatchboxCollectionNotFoundError())

    response = test_client.get("/collections/nonexistent")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_create_collection(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test creating a new collection."""
    test_client, mock_backend, _ = api_client_and_mocks

    new_collection = Collection(name="new_collection", versions={})
    mock_backend.create_collection = Mock(return_value=new_collection)

    response = test_client.post("/collections", json={"name": "new_collection"})

    assert response.status_code == 201
    assert response.json()["name"] == "new_collection"
    mock_backend.create_collection.assert_called_once_with(name="new_collection")


def test_create_collection_missing_name(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test creating a collection without providing a name."""
    test_client, _, _ = api_client_and_mocks

    response = test_client.post("/collections", json={})

    assert response.status_code == 422
    assert "Collection name is required" in response.json()["detail"]


def test_create_collection_already_exists(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test creating a collection that already exists."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.create_collection = Mock(side_effect=MatchboxCollectionAlreadyExists())

    response = test_client.post("/collections", json={"name": "existing_collection"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Collection already exists"


def test_delete_collection(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test deleting a collection with confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_collection = Mock()

    response = test_client.delete(
        "/collections/test_collection", params={"certain": True}
    )

    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]
    mock_backend.delete_collection.assert_called_once_with(
        name="test_collection", certain=True
    )


def test_delete_collection_needs_confirmation(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test deleting a collection that requires confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_collection = Mock(
        side_effect=MatchboxDeletionNotConfirmed(children=["v1", "v2"])
    )

    response = test_client.delete("/collections/test_collection")

    assert response.status_code == 409
    error_message = response.json()["detail"]
    assert "v1" in error_message and "v2" in error_message


def test_delete_collection_404(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test deleting a non-existent collection."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_collection = Mock(side_effect=MatchboxCollectionNotFoundError())

    response = test_client.delete("/collections/nonexistent", params={"certain": True})

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


# Version management tests


def test_list_versions(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test listing versions for a collection."""
    test_client, mock_backend, _ = api_client_and_mocks

    source_resolution = source_factory().source.to_resolution()
    versions_data = {"v1": [source_resolution], "v2": []}

    mock_backend.get_collection = Mock(return_value=versions_data)

    response = test_client.get("/collections/test_collection/versions")

    assert response.status_code == 200
    assert "v1" in response.json()
    assert "v2" in response.json()
    mock_backend.get_collection.assert_called_once_with(name="test_collection")


def test_get_version(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test retrieving a specific version."""
    test_client, mock_backend, _ = api_client_and_mocks

    source_resolution = source_factory().source.to_resolution()
    version_resolutions = [source_resolution]

    mock_backend.get_version = Mock(return_value=version_resolutions)

    response = test_client.get("/collections/test_collection/versions/v1")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["name"] == source_resolution.name
    mock_backend.get_version.assert_called_once_with(
        collection="test_collection", name="v1"
    )


def test_get_version_404(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test retrieving a non-existent version."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_version = Mock(side_effect=MatchboxCollectionNotFoundError())

    response = test_client.get("/collections/nonexistent/versions/v1")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_create_version(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test creating a new version."""
    test_client, mock_backend, _ = api_client_and_mocks

    new_version = Version(name="v2", is_default=False, is_mutable=True, resolutions=[])
    mock_backend.create_version = Mock(return_value=new_version)

    response = test_client.post(
        "/collections/test_collection/versions", json={"name": "v2"}
    )

    assert response.status_code == 201
    assert response.json()["name"] == "v2"
    assert response.json()["is_mutable"] is True
    mock_backend.create_version.assert_called_once_with(
        collection="test_collection", name="v2"
    )


def test_create_version_missing_name(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test creating a version without providing a name."""
    test_client, _, _ = api_client_and_mocks

    response = test_client.post("/collections/test_collection/versions", json={})

    assert response.status_code == 422
    assert "Version name is required" in response.json()["detail"]


def test_create_version_collection_not_found(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test creating a version in a non-existent collection."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.create_version = Mock(side_effect=MatchboxCollectionNotFoundError())

    response = test_client.post(
        "/collections/nonexistent/versions", json={"name": "v1"}
    )

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_create_version_already_exists(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test creating a version that already exists."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.create_version = Mock(side_effect=MatchboxVersionAlreadyExists())

    response = test_client.post(
        "/collections/test_collection/versions", json={"name": "v1"}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Version already exists"


def test_set_version_mutable(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test setting version mutability."""
    test_client, mock_backend, _ = api_client_and_mocks

    updated_version = Version(
        name="v1", is_default=False, is_mutable=False, resolutions=[]
    )
    mock_backend.set_version_mutable = Mock(return_value=updated_version)

    response = test_client.patch(
        "/collections/test_collection/versions/v1/mutable", json={"mutable": False}
    )

    assert response.status_code == 200
    assert response.json()["is_mutable"] is False
    mock_backend.set_version_mutable.assert_called_once_with(
        collection="test_collection", name="v1", mutable=False
    )


def test_set_version_mutable_missing_flag(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test setting version mutability without providing the flag."""
    test_client, _, _ = api_client_and_mocks

    response = test_client.patch(
        "/collections/test_collection/versions/v1/mutable", json={}
    )

    assert response.status_code == 422
    assert "Mutable flag is required" in response.json()["detail"]


def test_set_version_default(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test setting version as default."""
    test_client, mock_backend, _ = api_client_and_mocks

    updated_version = Version(
        name="v1", is_default=True, is_mutable=False, resolutions=[]
    )
    mock_backend.set_version_default = Mock(return_value=updated_version)

    response = test_client.patch(
        "/collections/test_collection/versions/v1/default", json={"default": True}
    )

    assert response.status_code == 200
    assert response.json()["is_default"] is True
    mock_backend.set_version_default.assert_called_once_with(
        collection="test_collection", name="v1", default=True
    )


def test_set_version_default_missing_flag(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test setting version default without providing the flag."""
    test_client, _, _ = api_client_and_mocks

    response = test_client.patch(
        "/collections/test_collection/versions/v1/default", json={}
    )

    assert response.status_code == 422
    assert "Default flag is required" in response.json()["detail"]


@pytest.mark.parametrize(
    "endpoint,payload",
    [("mutable", {"mutable": True}), ("default", {"default": False})],
)
def test_version_patch_endpoints_404(
    endpoint: str,
    payload: dict[str, bool],
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test 404 responses for version PATCH endpoints when resource doesn't exist."""
    test_client, mock_backend, _ = api_client_and_mocks
    method_name = f"set_version_{endpoint}"
    mock_method = getattr(mock_backend, method_name)
    mock_method.side_effect = MatchboxCollectionNotFoundError()

    response = test_client.patch(
        f"/collections/nonexistent/versions/v1/{endpoint}", json=payload
    )

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_delete_version(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test deleting a version with confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_version = Mock()

    response = test_client.delete(
        "/collections/test_collection/versions/v1", params={"certain": True}
    )

    assert response.status_code == 200
    assert response.json() == OKMessage().model_dump()
    mock_backend.delete_version.assert_called_once_with(
        collection="test_collection", name="v1", certain=True
    )


def test_delete_version_needs_confirmation(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test deleting a version that requires confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_version = Mock(
        side_effect=MatchboxDeletionNotConfirmed(
            children=["resolution1", "resolution2"]
        )
    )

    response = test_client.delete("/collections/test_collection/versions/v1")

    assert response.status_code == 409
    error_message = response.json()["detail"]
    assert "resolution1" in error_message and "resolution2" in error_message


def test_delete_version_404(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test deleting a non-existent version."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_version = Mock(side_effect=MatchboxCollectionNotFoundError())

    response = test_client.delete(
        "/collections/nonexistent/versions/v1", params={"certain": True}
    )

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


# Resolution listing test


def test_list_resolutions(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test listing resolutions in a version."""
    test_client, mock_backend, _ = api_client_and_mocks

    source_resolution = source_factory().source.to_resolution()
    model_resolution = model_factory().model.to_resolution()
    resolutions = [source_resolution, model_resolution]

    mock_backend.get_version = Mock(return_value=resolutions)

    response = test_client.get("/collections/test_collection/versions/v1/resolutions")

    assert response.status_code == 200
    assert len(response.json()) == 2
    # Verify we can deserialise the resolutions
    for resolution_data in response.json():
        assert "name" in resolution_data
        assert "resolution_type" in resolution_data

    mock_backend.get_version.assert_called_once_with(
        collection="test_collection", name="v1"
    )


def test_list_resolutions_404(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test listing resolutions for a non-existent collection/version."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_version = Mock(side_effect=MatchboxCollectionNotFoundError())

    response = test_client.get("/collections/nonexistent/versions/v1/resolutions")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION
