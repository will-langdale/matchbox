import asyncio
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import ANY, Mock, call, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from matchbox.common.arrow import SCHEMA_MB_IDS, table_to_buffer
from matchbox.common.dtos import BackendRetrievableType, UploadStatus
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.sources import source_factory
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import hash_to_base64
from matchbox.common.sources import Match, Source, SourceAddress
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

# def test_list_sources():
#     response = client.get("/sources")
#     assert response.status_code == 200


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_source(get_backend):
    dummy_source = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar"), db_pk="pk"
    )
    mock_backend = Mock()
    mock_backend.get_source = Mock(return_value=dummy_source)
    get_backend.return_value = mock_backend

    response = client.get(f"/sources/{hash_to_base64(b'bar')}/foo")
    assert response.status_code == 200
    assert Source.model_validate(response.json())


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_source_404(get_backend):
    mock_backend = Mock()
    mock_backend.get_source = Mock(side_effect=MatchboxSourceNotFoundError)
    get_backend.return_value = mock_backend

    response = client.get(f"/sources/{hash_to_base64(b'bar')}/foo")
    assert response.status_code == 404
    assert response.json()["entity"] == BackendRetrievableType.SOURCE


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
    assert response.json()["status"] == "queued"  # Updated to check for queued status
    # Check both status updates were called in correct order
    assert metadata_store.update_status.call_args_list == [
        call(update_id, "queued"),
    ]
    mock_backend.index.assert_not_called()  # Index happens in background
    mock_add_task.assert_called_once()  # Verify background task was queued


@patch("matchbox.server.base.BackendManager.get_backend")
@patch("matchbox.server.api.routes.metadata_store")
def test_upload_status_check(metadata_store: Mock, get_backend: Mock):
    """Test checking status of an upload using the status endpoint."""
    # Setup store with a processing entry
    store = MetadataStore()
    dummy_source = source_factory()
    update_id = store.cache_source(dummy_source.source)
    store.update_status(update_id, "processing")

    metadata_store.get.side_effect = store.get
    metadata_store.update_status.side_effect = store.update_status

    # Check status using GET endpoint
    response = client.get(f"/upload/{update_id}/status")

    # Should return current status
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    metadata_store.update_status.assert_not_called()


@patch("matchbox.server.base.BackendManager.get_backend")
@patch("matchbox.server.api.routes.metadata_store")
def test_upload_already_processing(metadata_store: Mock, get_backend: Mock):
    """Test attempting to upload when status is already processing."""
    # Setup store with a processing entry
    store = MetadataStore()
    dummy_source = source_factory()
    update_id = store.cache_source(dummy_source.source)
    store.update_status(update_id, "processing")

    metadata_store.get.side_effect = store.get

    # Attempt upload
    response = client.post(
        f"/upload/{update_id}",
        files={"file": ("test.parquet", b"dummy data", "application/octet-stream")},
    )

    # Should return 400 with current status
    assert response.status_code == 400
    assert response.json()["status"] == "processing"


@patch("matchbox.server.api.routes.metadata_store")
def test_upload_already_queued(metadata_store: Mock):
    """Test attempting to upload when status is already queued."""
    # Setup store with a queued entry
    store = MetadataStore()
    dummy_source = source_factory()
    update_id = store.cache_source(dummy_source.source)
    store.update_status(update_id, "queued")

    metadata_store.get.side_effect = store.get

    # Attempt upload
    response = client.post(
        f"/upload/{update_id}",
        files={"file": ("test.parquet", b"dummy data", "application/octet-stream")},
    )

    # Should return 400 with current status
    assert response.status_code == 400
    assert response.json()["status"] == "queued"


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


@patch("matchbox.server.api.routes.metadata_store")
def test_status_check_not_found(metadata_store: Mock):
    """Test checking status for non-existent upload ID."""
    metadata_store.get.return_value = None

    response = client.get("/upload/nonexistent-id/status")

    assert response.status_code == 400
    assert response.json()["status"] == "failed"
    assert "not found or expired" in response.json()["details"].lower()


@pytest.mark.asyncio
@patch("matchbox.server.base.BackendManager.get_backend")
async def test_complete_upload_process(get_backend: Mock, s3: S3Client):
    """Test the complete upload process from source creation through processing."""
    # Setup the backend
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    mock_backend.index = Mock(return_value=None)
    get_backend.return_value = mock_backend

    # Create test bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Create test data
    dummy_source = source_factory()

    # Step 1: Add source
    response = client.post("/sources", json=dummy_source.source.model_dump())
    assert response.status_code == 200
    upload_id = response.json()["id"]
    assert response.json()["status"] == "awaiting_upload"

    # Step 2: Upload file with real background tasks
    response = client.post(
        f"/upload/{upload_id}",
        files={
            "file": (
                "hashes.parquet",
                table_to_buffer(dummy_source.data_hashes),
                "application/octet-stream",
            ),
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"

    # Step 3: Poll status until complete or timeout
    max_attempts = 10
    current_attempt = 0
    while current_attempt < max_attempts:
        response = client.get(f"/upload/{upload_id}/status")
        assert response.status_code == 200

        status = response.json()["status"]
        if status == "complete":
            break
        elif status == "failed":
            pytest.fail(f"Upload failed: {response.json().get('details')}")
        elif status in ["processing", "queued"]:
            await asyncio.sleep(0.1)  # Small delay between polls
        else:
            pytest.fail(f"Unexpected status: {status}")

        current_attempt += 1

    assert current_attempt < max_attempts, (
        "Timed out waiting for processing to complete"
    )
    assert status == "complete"

    # Verify backend.index was called with correct arguments
    mock_backend.index.assert_called_once()
    call_args = mock_backend.index.call_args
    assert call_args[1]["source"] == dummy_source.source  # Check source matches
    assert call_args[1]["data_hashes"].equals(dummy_source.data_hashes)  # Check data


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
def test_match(get_backend: Mock):
    # Mock backend
    mock_matches = [
        Match(
            cluster=1,
            source=SourceAddress(full_name="foo", warehouse_hash=b"foo"),
            source_id={"1"},
            target=SourceAddress(full_name="bar", warehouse_hash=b"bar"),
            target_id={"a"},
        )
    ]
    mock_backend = Mock()
    mock_backend.match = Mock(return_value=mock_matches)
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = client.get(
        "/match",
        params={
            "target_full_names": ["foo"],
            "target_warehouse_hashes_b64": [hash_to_base64(b"foo")],
            "source_full_name": "bar",
            "source_warehouse_hash_b64": hash_to_base64(b"bar"),
            "source_pk": 1,
            "resolution_name": "res",
            "threshold": 50,
        },
    )

    # Check response
    assert response.status_code == 200
    [Match.model_validate(m) for m in response.json()]


@patch("matchbox.server.base.BackendManager.get_backend")
def test_match_404_resolution(get_backend: Mock):
    # Mock backend
    mock_backend = Mock()
    mock_backend.match = Mock(side_effect=MatchboxResolutionNotFoundError())
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = client.get(
        "/match",
        params={
            "target_full_names": ["foo"],
            "target_warehouse_hashes_b64": [hash_to_base64(b"foo")],
            "source_full_name": "bar",
            "source_warehouse_hash_b64": hash_to_base64(b"bar"),
            "source_pk": 1,
            "resolution_name": "res",
        },
    )

    # Check response
    assert response.status_code == 404
    assert response.json()["entity"] == BackendRetrievableType.RESOLUTION


@patch("matchbox.server.base.BackendManager.get_backend")
def test_match_404_source(get_backend: Mock):
    # Mock backend
    mock_backend = Mock()
    mock_backend.match = Mock(side_effect=MatchboxSourceNotFoundError())
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = client.get(
        "/match",
        params={
            "target_full_names": ["foo"],
            "target_warehouse_hashes_b64": [hash_to_base64(b"foo")],
            "source_full_name": "bar",
            "source_warehouse_hash_b64": hash_to_base64(b"bar"),
            "source_pk": 1,
            "resolution_name": "res",
        },
    )

    # Check response
    assert response.status_code == 404
    assert response.json()["entity"] == BackendRetrievableType.SOURCE


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
