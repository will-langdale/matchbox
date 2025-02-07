from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import ANY, Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from matchbox.common.arrow import SCHEMA_MB_IDS, table_to_buffer
from matchbox.common.dtos import UploadStatus
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.sources import source_factory
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import hash_to_base64
from matchbox.server import app
from matchbox.server.api.cache import MetadataStore
from matchbox.server.base import MatchboxDBAdapter

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any

client = TestClient(app)


def test_healthcheck():
    """Test the healthcheck endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


@patch("matchbox.server.base.BackendManager.get_backend")
def test_count_all_backend_items(get_backend):
    """Test the unparameterised entity counting endpoint."""
    entity_counts = {
        "datasets": 1,
        "models": 2,
        "data": 3,
        "clusters": 4,
        "creates": 5,
        "merges": 6,
        "proposes": 7,
    }
    mock_backend = Mock()
    for e, c in entity_counts.items():
        mock_e = Mock()
        mock_e.count = Mock(return_value=c)
        setattr(mock_backend, e, mock_e)
    get_backend.return_value = mock_backend

    response = client.get("/testing/count")
    assert response.status_code == 200
    assert response.json() == {"entities": entity_counts}


@patch("matchbox.server.base.BackendManager.get_backend")
def test_count_backend_item(get_backend: MatchboxDBAdapter):
    """Test the parameterised entity counting endpoint."""
    mock_backend = Mock()
    mock_backend.models.count = Mock(return_value=20)
    get_backend.return_value = mock_backend

    response = client.get("/testing/count", params={"entity": "models"})
    assert response.status_code == 200
    assert response.json() == {"entities": {"models": 20}}


# def test_clear_backend():
#     response = client.post("/testing/clear")
#     assert response.status_code == 200


@patch("matchbox.server.base.BackendManager.get_backend")
def test_add_source(get_backend: Mock):
    """Test the source addition endpoint."""
    # Setup
    mock_backend = Mock()
    mock_backend.index = Mock(return_value=None)
    get_backend.return_value = mock_backend

    dummy_source = source_factory()

    # Make request
    response = client.post("/sources", json=dummy_source.source.model_dump())

    # Validate response
    assert UploadStatus.model_validate(response.json())
    assert response.status_code == 200, response.json()
    assert response.json()["status"] == "awaiting_upload"
    assert response.json().get("id") is not None
    mock_backend.index.assert_not_called()


@patch("matchbox.server.base.BackendManager.get_backend")
@patch("matchbox.server.api.routes.metadata_store")
@patch("matchbox.server.api.routes.BackgroundTasks.add_task")
def test_source_upload(
    mock_add_task: Mock, metadata_store: Mock, get_backend: Mock, s3: S3Client
):
    """Test uploading a file, happy path."""
    # Setup
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    mock_backend.index = Mock(return_value=None)
    get_backend.return_value = mock_backend
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    dummy_source = source_factory()

    # Mock the metadata store
    store = MetadataStore()
    update_id = store.cache_source(dummy_source.source)
    metadata_store.get.side_effect = store.get
    metadata_store.update_status.side_effect = store.update_status

    # Make request with mocked background task
    response = client.post(
        f"/upload/{update_id}",
        files={
            "file": (
                "hashes.parquet",
                table_to_buffer(dummy_source.data_hashes),
                "application/octet-stream",
            ),
        },
    )

    # Validate response
    assert UploadStatus.model_validate(response.json())
    assert response.status_code == 200, response.json()
    assert response.json()["status"] == "processing"
    metadata_store.update_status.assert_called_with(update_id, "processing")
    mock_backend.index.assert_not_called()  # Index happens in background
    mock_add_task.assert_called_once()  # Verify background task was queued


@patch("matchbox.server.base.BackendManager.get_backend")
@patch("matchbox.server.api.routes.metadata_store")
def test_source_upload_status_check(metadata_store: Mock, _: Mock):
    """Test checking status of an upload in progress."""
    # Setup store with a processing entry
    store = MetadataStore()
    dummy_source = source_factory()
    update_id = store.cache_source(dummy_source.source)
    store.update_status(update_id, "processing")

    metadata_store.get.side_effect = store.get
    metadata_store.update_status.side_effect = store.update_status

    # Check status
    response = client.post(f"/upload/{update_id}")

    # Should return current status without starting new upload
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    metadata_store.update_status.assert_not_called()


@patch("matchbox.server.base.BackendManager.get_backend")
@patch("matchbox.server.api.routes.metadata_store")
@patch("matchbox.server.api.routes.BackgroundTasks.add_task")
def test_source_upload_wrong_schema(
    mock_add_task: Mock, metadata_store: Mock, get_backend: Mock, s3: S3Client
):
    """Test uploading a file with wrong schema."""
    # Setup
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    get_backend.return_value = mock_backend

    # Create source with results schema instead of index
    dummy_source = source_factory()

    # Setup store
    store = MetadataStore()
    update_id = store.cache_source(dummy_source.source)
    metadata_store.get.side_effect = store.get
    metadata_store.update_status.side_effect = store.update_status

    # Make request with actual data instead of the hashes -- wrong schema
    response = client.post(
        f"/upload/{update_id}",
        files={
            "file": (
                "hashes.parquet",
                table_to_buffer(dummy_source.data),
                "application/octet-stream",
            ),
        },
    )

    # Should fail before background task starts
    assert response.status_code == 400
    assert response.json()["status"] == "failed"
    assert "schema mismatch" in response.json()["details"].lower()
    metadata_store.update_status.assert_called_with(update_id, "failed", details=ANY)
    mock_add_task.assert_not_called()  # Background task should not be queued


# def test_list_sources():
#     response = client.get("/sources")
#     assert response.status_code == 200

# def test_get_source():
#     response = client.get("/sources/test_source")
#     assert response.status_code == 200

# def test_list_models():
#     response = client.get("/models")
#     assert response.status_code == 200

# def test_get_resolution():
#     response = client.get("/models/test_resolution")
#     assert response.status_code == 200

# def test_add_model():
#     response = client.post("/models")
#     assert response.status_code == 200

# def test_delete_model():
#     response = client.delete("/models/test_model")
#     assert response.status_code == 200

# def test_get_results():
#     response = client.get("/models/test_model/results")
#     assert response.status_code == 200

# def test_set_results():
#     response = client.post("/models/test_model/results")
#     assert response.status_code == 200

# def test_get_truth():
#     response = client.get("/models/test_model/truth")
#     assert response.status_code == 200

# def test_set_truth():
#     response = client.post("/models/test_model/truth")
#     assert response.status_code == 200

# def test_get_ancestors():
#     response = client.get("/models/test_model/ancestors")
#     assert response.status_code == 200

# def test_get_ancestors_cache():
#     response = client.get("/models/test_model/ancestors_cache")
#     assert response.status_code == 200

# def test_set_ancestors_cache():
#     response = client.post("/models/test_model/ancestors_cache")
#     assert response.status_code == 200

# def test_query():
#     response = client.get("/query")
#     assert response.status_code == 200


@patch("matchbox.server.base.BackendManager.get_backend")
def test_query(get_backend: Mock):
    # Mock backend
    mock_backend = Mock()
    mock_backend.query = Mock(
        return_value=pa.Table.from_pylist(
            [
                {"source_pk": "a", "id": 1},
                {"source_pk": "b", "id": 2},
            ],
            schema=SCHEMA_MB_IDS,
        )
    )
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = client.get(
        "/query",
        params={
            "full_name": "foo",
            "warehouse_hash_b64": hash_to_base64(b"bar"),
        },
    )

    # Process response
    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)

    # Check response
    assert response.status_code == 200
    assert table.schema.equals(SCHEMA_MB_IDS)


@patch("matchbox.server.base.BackendManager.get_backend")
def test_query_404_resolution(get_backend: Mock):
    # Mock backend
    mock_backend = Mock()
    mock_backend.query = Mock(side_effect=MatchboxResolutionNotFoundError())
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = client.get(
        "/query",
        params={
            "full_name": "foo",
            "warehouse_hash_b64": hash_to_base64(b"bar"),
        },
    )

    # Check response
    assert response.status_code == 404


@patch("matchbox.server.base.BackendManager.get_backend")
def test_query_404_source(get_backend: Mock):
    # Mock backend
    mock_backend = Mock()
    mock_backend.query = Mock(side_effect=MatchboxSourceNotFoundError())
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = client.get(
        "/query",
        params={
            "full_name": "foo",
            "warehouse_hash_b64": hash_to_base64(b"bar"),
        },
    )

    # Check response
    assert response.status_code == 404


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_resolution_graph(
    get_backend: MatchboxDBAdapter, resolution_graph: ResolutionGraph
):
    """Test the resolution graph report endpoint."""
    mock_backend = Mock()
    mock_backend.get_resolution_graph = Mock(return_value=resolution_graph)
    get_backend.return_value = mock_backend

    response = client.get("/report/resolutions")
    assert response.status_code == 200
    assert ResolutionGraph.model_validate(response.json())
