from importlib.metadata import version
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import ANY, Mock, call, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from matchbox.client.authorisation import (
    generate_json_web_token,
)
from matchbox.common.arrow import (
    SCHEMA_QUERY,
    SCHEMA_QUERY_WITH_PROBABILITIES,
    table_to_buffer,
)
from matchbox.common.dtos import (
    BackendResourceType,
    LoginAttempt,
    LoginResult,
    OKMessage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.sources import source_factory
from matchbox.common.graph import ResolutionGraph
from matchbox.common.sources import Match
from matchbox.server.api.dependencies import backend, upload_tracker
from matchbox.server.api.main import app
from matchbox.server.api.uploads import UploadTracker, process_upload

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


# General


def test_healthcheck(test_client: TestClient):
    """Test the healthcheck endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    response = OKMessage.model_validate(response.json())
    assert response.status == "OK"
    assert response.version == version("matchbox-db")


def test_login(test_client: TestClient):
    """Test the login endpoint."""
    mock_backend = Mock()
    mock_backend.login = Mock(return_value=1)

    response = test_client.post(
        "/login", json=LoginAttempt(user_name="alice").model_dump()
    )

    assert response.status_code == 200
    response = LoginResult.model_validate(response.json())
    assert response.user_id == 1


@patch("matchbox.server.api.main.BackgroundTasks.add_task")
def test_upload(
    mock_add_task: Mock,
    s3: S3Client,
    test_client: TestClient,
):
    """Test uploading a file, happy path."""
    # Setup
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    mock_backend.index = Mock(return_value=None)
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    source_testkit = source_factory()

    # Mock the metadata store
    mock_upload_tracker = Mock()
    tracker = UploadTracker()
    update_id = tracker.add_source(source_testkit.source_config)
    mock_upload_tracker.get.side_effect = tracker.get
    mock_upload_tracker.update_status.side_effect = tracker.update_status

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend
    app.dependency_overrides[upload_tracker] = lambda: mock_upload_tracker

    # Make request with mocked background task
    response = test_client.post(
        f"/upload/{update_id}",
        files={
            "file": (
                "hashes.parquet",
                table_to_buffer(source_testkit.data_hashes),
                "application/octet-stream",
            ),
        },
    )

    # Validate response
    assert UploadStatus.model_validate(response.json())
    assert response.status_code == 202, response.json()
    assert response.json()["status"] == "queued"  # Updated to check for queued status
    # Check both status updates were called in correct order
    assert mock_upload_tracker.update_status.call_args_list == [
        call(update_id, "queued"),
    ]
    mock_backend.index.assert_not_called()  # Index happens in background
    mock_add_task.assert_called_once()  # Verify background task was queued


@patch("matchbox.server.api.main.BackgroundTasks.add_task")
def test_upload_wrong_schema(
    mock_add_task: Mock,
    s3: S3Client,
    test_client: TestClient,
):
    """Test uploading a file with wrong schema."""
    # Setup
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"

    # Create source with results schema instead of index
    source_testkit = source_factory()

    # Setup store
    mock_upload_tracker = Mock()
    tracker = UploadTracker()
    update_id = tracker.add_source(source_testkit.source_config)
    mock_upload_tracker.get.side_effect = tracker.get
    mock_upload_tracker.update_status.side_effect = tracker.update_status

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend
    app.dependency_overrides[upload_tracker] = lambda: mock_upload_tracker

    # Make request with actual data instead of the hashes -- wrong schema
    response = test_client.post(
        f"/upload/{update_id}",
        files={
            "file": (
                "hashes.parquet",
                table_to_buffer(source_testkit.data),
                "application/octet-stream",
            ),
        },
    )

    # Should fail before background task starts
    assert response.status_code == 400
    assert response.json()["status"] == "failed"
    assert "schema mismatch" in response.json()["details"].lower()
    mock_upload_tracker.update_status.assert_called_with(
        update_id, "failed", details=ANY
    )
    mock_add_task.assert_not_called()  # Background task should not be queued


def test_upload_status_check(test_client: TestClient):
    """Test checking status of an upload using the status endpoint."""
    # Setup store with a processing entry
    mock_upload_tracker = Mock()
    tracker = UploadTracker()
    source_testkit = source_factory()
    update_id = tracker.add_source(source_testkit.source_config)
    tracker.update_status(update_id, "processing")

    mock_upload_tracker.get.side_effect = tracker.get
    mock_upload_tracker.update_status.side_effect = tracker.update_status

    # Override app dependencies with mocks
    app.dependency_overrides[upload_tracker] = lambda: mock_upload_tracker

    # Check status using GET endpoint
    response = test_client.get(f"/upload/{update_id}/status")

    # Should return current status
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    mock_upload_tracker.update_status.assert_not_called()


def test_upload_already_processing(test_client: TestClient):
    """Test attempting to upload when status is already processing."""
    # Setup store with a processing entry
    mock_upload_tracker = Mock()
    tracker = UploadTracker()
    source_testkit = source_factory()
    update_id = tracker.add_source(source_testkit.source_config)
    tracker.update_status(update_id, "processing")

    mock_upload_tracker.get.side_effect = tracker.get

    # Override app dependencies with mocks
    app.dependency_overrides[upload_tracker] = lambda: mock_upload_tracker

    # Attempt upload
    response = test_client.post(
        f"/upload/{update_id}",
        files={"file": ("test.parquet", b"dummy data", "application/octet-stream")},
    )

    # Should return 400 with current status
    assert response.status_code == 400
    assert response.json()["status"] == "processing"


def test_upload_already_queued(test_client: TestClient):
    """Test attempting to upload when status is already queued."""
    # Setup store with a queued entry
    mock_upload_tracker = Mock()
    tracker = UploadTracker()
    source_testkit = source_factory()
    update_id = tracker.add_source(source_testkit.source_config)
    tracker.update_status(update_id, "queued")

    mock_upload_tracker.get.side_effect = tracker.get

    # Override app dependencies with mocks
    app.dependency_overrides[upload_tracker] = lambda: mock_upload_tracker

    # Attempt upload
    response = test_client.post(
        f"/upload/{update_id}",
        files={"file": ("test.parquet", b"dummy data", "application/octet-stream")},
    )

    # Should return 400 with current status
    assert response.status_code == 400
    assert response.json()["status"] == "queued"


def test_status_check_not_found(test_client: TestClient):
    """Test checking status for non-existent upload ID."""
    mock_upload_tracker = Mock()
    mock_upload_tracker.get.return_value = None

    # Override app dependencies with mocks
    app.dependency_overrides[upload_tracker] = lambda: mock_upload_tracker

    response = test_client.get("/upload/nonexistent-id/status")

    assert response.status_code == 400
    assert response.json()["status"] == "failed"
    assert "not found or expired" in response.json()["details"].lower()


def test_process_upload_deletes_file_on_failure(s3: S3Client):
    """Test that files are deleted from S3 even when processing fails."""
    # Setup
    bucket = "test-bucket"
    test_key = "test-upload-id.parquet"
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.index = Mock(side_effect=ValueError("Simulated processing failure"))

    s3.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Add parquet to S3 and verify
    source_testkit = source_factory()
    buffer = table_to_buffer(source_testkit.data_hashes)
    s3.put_object(Bucket=bucket, Key=test_key, Body=buffer)

    assert s3.head_object(Bucket=bucket, Key=test_key)

    # Setup metadata store with test data
    tracker = UploadTracker()
    upload_id = tracker.add_source(source_testkit.source_config)
    tracker.update_status(upload_id, "awaiting_upload")

    # Run the process, expecting it to fail
    with pytest.raises(MatchboxServerFileError) as excinfo:
        process_upload(
            backend=mock_backend,
            upload_id=upload_id,
            bucket=bucket,
            key=test_key,
            tracker=tracker,
            heartbeat_seconds=10,
        )

    assert "Simulated processing failure" in str(excinfo.value)

    # Check that the status was updated to failed
    status = tracker.get(upload_id).status
    assert status.status == "failed", f"Expected status 'failed', got '{status.status}'"

    # Verify file was deleted despite the failure
    with pytest.raises(ClientError) as excinfo:
        s3.head_object(Bucket=bucket, Key=test_key)
    assert "404" in str(excinfo.value) or "NoSuchKey" in str(excinfo.value), (
        f"File was not deleted: {str(excinfo.value)}"
    )


# Retrieval


def test_query(test_client: TestClient):
    # Mock backend
    mock_backend = Mock()
    mock_backend.query = Mock(
        return_value=pa.Table.from_pylist(
            [
                {"keys": "a", "id": 1},
                {"keys": "b", "id": 2},
            ],
            schema=SCHEMA_QUERY,
        )
    )

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Hit endpoint
    response = test_client.get(
        "/query",
        params={"source": "foo", "return_leaf_id": False},
    )

    # Process response
    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)

    # Check response
    assert response.status_code == 200
    assert table.schema.equals(SCHEMA_QUERY)


def test_query_with_probabilities(test_client: TestClient):
    """Test query endpoint with get_probabilities parameter."""
    # Mock backend
    mock_backend = Mock()
    mock_backend.query = Mock(
        return_value=pa.Table.from_pylist(
            [
                {"id": 1, "key": "a", "probability": 100},
                {"id": 2, "key": "b", "probability": 95},
            ],
            schema=SCHEMA_QUERY_WITH_PROBABILITIES,
        )
    )

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Hit endpoint with get_probabilities=True
    response = test_client.get(
        "/query",
        params={"source": "foo", "return_leaf_id": False, "get_probabilities": True},
    )

    # Process response
    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)

    # Check response
    assert response.status_code == 200
    assert table.schema.equals(SCHEMA_QUERY_WITH_PROBABILITIES)
    assert "probability" in table.column_names


def test_query_404_resolution(test_client: TestClient):
    mock_backend = Mock()
    mock_backend.query = Mock(side_effect=MatchboxResolutionNotFoundError())

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Hit endpoint
    response = test_client.get(
        "/query",
        params={"source": "foo", "resolution": "bar", "return_leaf_id": True},
    )

    # Check response
    assert response.status_code == 404


def test_query_404_source(test_client: TestClient):
    mock_backend = Mock()
    mock_backend.query = Mock(side_effect=MatchboxSourceNotFoundError())

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Hit endpoint
    response = test_client.get(
        "/query",
        params={"source": "foo", "return_leaf_id": True},
    )

    # Check response
    assert response.status_code == 404


def test_match(test_client: TestClient):
    mock_backend = Mock()
    mock_matches = [
        Match(
            cluster=1,
            source="foo",
            source_id={"1"},
            target="bar",
            target_id={"a"},
        )
    ]
    mock_backend.match = Mock(return_value=mock_matches)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Hit endpoint
    response = test_client.get(
        "/match",
        params={
            "targets": "foo",
            "source": "bar",
            "key": 1,
            "resolution": "res",
            "threshold": 50,
        },
    )

    # Check response
    assert response.status_code == 200
    [Match.model_validate(m) for m in response.json()]


def test_match_404_resolution(test_client: TestClient):
    mock_backend = Mock()
    mock_backend.match = Mock(side_effect=MatchboxResolutionNotFoundError())

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Hit endpoint
    response = test_client.get(
        "/match",
        params={
            "targets": ["foo"],
            "source": "bar",
            "key": 1,
            "resolution": "res",
        },
    )

    # Check response
    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_match_404_source(test_client: TestClient):
    mock_backend = Mock()
    mock_backend.match = Mock(side_effect=MatchboxSourceNotFoundError())

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Hit endpoint
    response = test_client.get(
        "/match",
        params={
            "targets": ["foo"],
            "source": "bar",
            "key": 1,
            "resolution": "res",
        },
    )

    # Check response
    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.SOURCE


# Admin


def test_count_all_backend_items(test_client: TestClient):
    """Test the unparameterised entity counting endpoint."""
    mock_backend = Mock()
    entity_counts = {
        "sources": 1,
        "models": 2,
        "data": 3,
        "clusters": 4,
        "creates": 5,
        "merges": 6,
        "proposes": 7,
    }
    for e, c in entity_counts.items():
        mock_e = Mock()
        mock_e.count = Mock(return_value=c)
        setattr(mock_backend, e, mock_e)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.get("/database/count")
    assert response.status_code == 200
    assert response.json() == {"entities": entity_counts}


def test_count_backend_item(test_client: TestClient):
    """Test the parameterised entity counting endpoint."""
    mock_backend = Mock()
    mock_backend.models.count = Mock(return_value=20)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.get("/database/count", params={"entity": "models"})
    assert response.status_code == 200
    assert response.json() == {"entities": {"models": 20}}


def test_clear_backend_ok(test_client: TestClient):
    mock_backend = Mock()
    mock_backend.clear = Mock()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.delete("/database", params={"certain": "true"})
    assert response.status_code == 200
    OKMessage.model_validate(response.json())


def test_clear_backend_errors(test_client: TestClient):
    mock_backend = Mock()
    mock_backend.clear = Mock(side_effect=MatchboxDeletionNotConfirmed)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.delete("/database")
    assert response.status_code == 409
    # We send some explanatory message
    assert response.content


def test_api_key_authorisation(test_client: TestClient):
    routes = [
        (test_client.post, "/upload/upload_id"),
        (test_client.post, "/sources"),
        (test_client.post, "/models"),
        (test_client.patch, "/models/name/truth"),
        (test_client.delete, "/resolutions/name"),
        (test_client.delete, "/database"),
    ]

    # Incorrect signature
    _, _, signature_b64 = test_client.headers["Authorization"].encode().split(b".")
    header_b64, payload_64, _ = (
        generate_json_web_token(sub="incorrect.user@email.com").encode().split(b".")
    )
    test_client.headers["Authorization"] = b".".join(
        [header_b64, payload_64, signature_b64]
    ).decode()

    for method, url in routes:
        response = method(url)
        assert response.status_code == 401
        assert response.content == b'"JWT invalid."'

    # Expired JWT
    with patch("matchbox.client.authorisation.EXPIRY_AFTER_X_HOURS", -2):
        test_client.headers["Authorization"] = generate_json_web_token(
            sub="test.user@email.com"
        )
        for method, url in routes:
            response = method(url)
            assert response.status_code == 401
            assert response.content == b'"JWT expired."'

    # Missing Authorization header
    test_client.headers.pop("Authorization")
    for method, url in routes:
        response = method(url)
        assert response.status_code == 401
        assert response.content == b'"JWT required but not provided."'


def test_get_resolution_graph(
    resolution_graph: ResolutionGraph, test_client: TestClient
):
    """Test the resolution graph report endpoint."""
    mock_backend = Mock()
    mock_backend.get_resolution_graph = Mock(return_value=resolution_graph)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.get("/report/resolutions")
    assert response.status_code == 200
    assert ResolutionGraph.model_validate(response.json())
