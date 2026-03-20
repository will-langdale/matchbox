from io import BytesIO
from unittest.mock import Mock, patch

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient
from pyarrow import parquet as pq

from matchbox.common.arrow import SCHEMA_CLUSTERS, table_to_buffer
from matchbox.common.dtos import (
    CRUDOperation,
    ErrorResponse,
    ResourceOperationStatus,
    StepPath,
    StepType,
    UploadInfo,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxLockError,
    MatchboxStepNotFoundError,
)
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import source_factory


def test_get_source(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    source_testkit = source_factory(name="foo").fake_run()
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_step = Mock(
        return_value=source_testkit.source.to_dto(),  # Second call (SOURCE)
    )

    response = test_client.get("/collections/default/runs/1/steps/foo")
    assert response.status_code == 200
    assert response.json()["step_type"] == "source"


def test_get_model(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    testkit = model_factory(name="test_model", description="test description")
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_step = Mock(return_value=testkit.fake_run().model.to_dto())

    response = test_client.get("/collections/default/runs/1/steps/test_model")

    assert response.status_code == 200
    assert response.json()["description"] == testkit.model.description


def test_get_step_404(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_step = Mock(side_effect=MatchboxStepNotFoundError())

    response = test_client.get("/collections/default/runs/1/steps/nonexistent")

    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxStepNotFoundError"


def test_insert_step(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Step metadata can be created."""
    testkit = model_factory(name="test_model").fake_run()
    test_client, mock_backend, _ = api_client_and_mocks

    response = test_client.post(
        "/collections/default/runs/1/steps/test_model",
        json=testkit.model.to_dto().model_dump(),
    )

    assert response.status_code == 201
    assert (
        response.json()
        == ResourceOperationStatus(
            success=True,
            target="Step default/1/test_model",
            operation=CRUDOperation.CREATE,
            details=None,
        ).model_dump()
    )

    mock_backend.create_step.assert_called_once_with(
        step=testkit.model.to_dto(),
        path=StepPath(name="test_model", collection="default", run=1),
    )


def test_insert_step_error(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.create_step = Mock(side_effect=Exception("Test error"))
    testkit = model_factory()

    response = test_client.post(
        "/collections/default/runs/1/steps/name",
        json=testkit.fake_run().model.to_dto().model_dump(),
    )

    assert response.status_code == 500
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxServerError"
    assert error.message.startswith("An internal server error occurred. Reference:")


def test_update_step(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Step metadata can be updated."""
    testkit = model_factory(name="test_model").fake_run()
    test_client, mock_backend, _ = api_client_and_mocks

    response = test_client.put(
        "/collections/default/runs/1/steps/test_model",
        json=testkit.model.to_dto().model_dump(),
    )

    assert response.status_code == 200
    assert (
        response.json()
        == ResourceOperationStatus(
            success=True,
            target="Step default/1/test_model",
            operation=CRUDOperation.UPDATE,
            details=None,
        ).model_dump()
    )

    mock_backend.update_step.assert_called_once_with(
        step=testkit.model.to_dto(),
        path=StepPath(name="test_model", collection="default", run=1),
    )

    # Errors are handled
    mock_backend.update_step = Mock(side_effect=Exception("Test error"))
    response = test_client.put(
        "/collections/default/runs/1/steps/name",
        json=testkit.model.to_dto().model_dump(),
    )

    assert response.status_code == 500
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxServerError"
    assert error.message.startswith("An internal server error occurred. Reference:")


# We can patch BackgroundTasks, since the api_client_and_mocks fixture
# ensures the API runs the task (not Celery)
@patch("matchbox.server.api.routers.collections.BackgroundTasks.add_task")
@patch("matchbox.server.api.routers.collections.file_to_s3")
def test_complete_upload_process(
    mock_add_task: Mock,
    mock_s3_upload: Mock,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test the complete step data upload from creation through processing."""
    test_client, mock_backend, _ = api_client_and_mocks

    # Create test data
    testkit = model_factory(model_type="linker").fake_run()
    collection = testkit.path.collection
    run = testkit.path.run

    # Mock insertion of data
    mock_backend.insert_model_data = Mock(return_value=None)
    # Mock retrieval of step metadata and data
    mock_backend.get_step = Mock(return_value=testkit.model.to_dto())
    mock_backend.get_model_data = Mock(return_value=testkit.scores.to_arrow())
    # Mock checking of status after upload
    mock_backend.get_step_stage = Mock(return_value=UploadStage.COMPLETE)

    # Step 1: Create step
    response = test_client.post(
        f"/collections/{collection}/runs/{run}/steps/{testkit.model.name}",
        json=testkit.model.to_dto().model_dump(),
    )
    assert response.status_code == 201
    step_post_info = ResourceOperationStatus.model_validate(response.json())
    assert step_post_info.success
    assert testkit.model.name in step_post_info.target

    # Step 2: Upload data file
    response = test_client.post(
        f"/collections/{collection}/runs/{run}/steps/{testkit.model.name}/data",
        files={
            "file": (
                "results.parquet",
                table_to_buffer(testkit.scores.to_arrow()),
                "application/octet-stream",
            ),
        },
    )
    assert response.status_code == 202
    data_post_info = ResourceOperationStatus.model_validate(response.json())
    upload_id = data_post_info.details
    assert mock_s3_upload.called
    assert mock_add_task.called

    # Step 3: Get upload status
    response = test_client.get(
        f"/collections/{collection}/runs/{run}/steps/{testkit.model.name}/data/status",
        params={"upload_id": upload_id},
    )
    assert response.status_code == 200
    upload_info = UploadInfo.model_validate(response.json())
    assert upload_info.stage == UploadStage.COMPLETE

    # Step 4: We can retrieve the results
    response = test_client.get(
        f"/collections/default/runs/1/steps/{testkit.model.name}/data"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"


@patch("matchbox.server.api.routers.collections.BackgroundTasks.add_task")
@patch("matchbox.server.api.routers.collections.file_to_s3")
def test_set_data_404(
    mock_add_task: Mock,
    mock_s3_upload: Mock,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test setting data for a non-existent step."""
    test_client, mock_backend, _ = api_client_and_mocks
    # Because we expect error when getting lock, we don't test
    # that lock was released
    mock_backend.lock_step_data = Mock(side_effect=MatchboxStepNotFoundError())

    response = test_client.post(
        "/collections/default/runs/1/steps/nonexistent-model/data",
        files={
            "file": (
                "hashes.parquet",
                table_to_buffer(source_factory().data_hashes),
                "application/octet-stream",
            ),
        },
    )

    # Correct error response
    assert response.status_code == 404
    error404 = ErrorResponse.model_validate(response.json())
    assert error404.exception_type == "MatchboxStepNotFoundError"
    # Upload doesn't proceed
    mock_s3_upload.assert_not_called()
    mock_add_task.assert_not_called()


@patch("matchbox.server.api.routers.collections.BackgroundTasks.add_task")
@patch("matchbox.server.api.routers.collections.file_to_s3")
def test_set_data_file_format(
    mock_add_task: Mock,
    mock_s3_upload: Mock,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test that file uploaded has Parquet magic bytes."""
    # Setup
    test_client, mock_backend, _ = api_client_and_mocks

    # Make request with mocked background task
    response = test_client.post(
        "/collections/default/runs/1/steps/step/data",
        files={
            "file": (
                "hashes.parquet",
                b"dummy\ndata",
                "application/octet-stream",
            ),
        },
    )

    # Correct error response
    assert response.status_code == 400
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxServerFileError"
    assert "invalid parquet" in error.message.lower()
    # Upload doesn't proceed
    mock_s3_upload.assert_not_called()
    mock_add_task.assert_not_called()
    mock_backend.unlock_step_data.assert_called()


@patch("matchbox.server.api.routers.collections.BackgroundTasks.add_task")
@patch("matchbox.server.api.routers.collections.file_to_s3")
def test_set_data_already_queued(
    mock_add_task: Mock,
    mock_s3_upload: Mock,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test attempting to upload when status is already queued."""
    test_client, mock_backend, _ = api_client_and_mocks
    # Because we expect error when getting lock, we don't test
    # that lock was released
    mock_backend.lock_step_data.side_effect = MatchboxLockError(
        "Upload already being processed."
    )

    response = test_client.post(
        "/collections/default/runs/1/steps/step/data",
        files={
            "file": (
                "hashes.parquet",
                table_to_buffer(source_factory().data_hashes),
                "application/octet-stream",
            ),
        },
    )

    # Correct error response
    assert response.status_code == 423
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxLockError"
    assert "Upload already being processed." in error.message
    # Upload doesn't proceed
    mock_s3_upload.assert_not_called()
    mock_add_task.assert_not_called()


def test_get_upload_status_404(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test getting upload status for a non-existent step."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_step_stage = Mock(side_effect=MatchboxStepNotFoundError())

    response = test_client.get(
        "/collections/default/runs/1/steps/nonexistent-model/data/status"
    )

    assert response.status_code == 404
    error404 = ErrorResponse.model_validate(response.json())
    assert error404.exception_type == "MatchboxStepNotFoundError"


def test_get_upload_status_gets_errors(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test getting failed upload status."""
    test_client, mock_backend, mock_tracker = api_client_and_mocks
    mock_backend.get_step_stage = Mock(return_value=UploadStage.READY)
    mock_tracker.set("upload_id", "error message")

    response = test_client.get(
        "/collections/default/runs/1/steps/nonexistent-model/data/status",
        params={"upload_id": "upload_id"},
    )

    assert response.status_code == 200
    info = UploadInfo.model_validate(response.json())
    assert info.error == "error message"


def test_get_results(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    testkit = model_factory()
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_step = Mock(return_value=Mock(step_type=StepType.MODEL))
    mock_backend.get_model_data = Mock(return_value=testkit.scores.to_arrow())

    response = test_client.get(
        f"/collections/default/runs/1/steps/{testkit.model.name}/data"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    mock_backend.get_model_data.assert_called_once()


def test_get_results_resolver(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    resolver_data = pa.Table.from_pylist(
        [{"parent_id": 1, "child_id": 1}],
        schema=SCHEMA_CLUSTERS,
    )

    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_step = Mock(return_value=Mock(step_type=StepType.RESOLVER))
    mock_backend.get_resolver_data = Mock(return_value=resolver_data)

    response = test_client.get("/collections/default/runs/1/steps/resolver/data")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    result_table = pq.read_table(BytesIO(response.content))
    assert result_table.schema.equals(SCHEMA_CLUSTERS)
    mock_backend.get_resolver_data.assert_called_once()


def test_get_results_rejects_source_step(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_step = Mock(return_value=Mock(step_type=StepType.SOURCE))

    response = test_client.get("/collections/default/runs/1/steps/source/data")

    assert response.status_code == 422
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxStepTypeError"


def test_delete_step(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test deletion of a step."""
    testkit = model_factory()
    test_client, _, _ = api_client_and_mocks
    response = test_client.delete(
        f"/collections/default/runs/1/steps/{testkit.model.name}",
        params={"certain": True},
    )

    assert response.status_code == 200
    assert (
        response.json()
        == ResourceOperationStatus(
            success=True,
            target=f"Step default/1/{testkit.model.name}",
            operation=CRUDOperation.DELETE,
            details=None,
        ).model_dump()
    )


def test_delete_step_needs_confirmation(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test deletion of a model that requires confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_step = Mock(
        side_effect=MatchboxDeletionNotConfirmed(children=["dedupe1", "dedupe2"])
    )

    testkit = model_factory()
    response = test_client.delete(
        f"/collections/default/runs/1/steps/{testkit.model.name}"
    )

    assert response.status_code == 409
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxDeletionNotConfirmed"
    assert "dedupe1" in error.message and "dedupe2" in error.message


@pytest.mark.parametrize("certain", [True, False])
def test_delete_step_404(
    certain: bool,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404 response when trying to delete a non-existent step."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_step.side_effect = MatchboxStepNotFoundError()

    response = test_client.delete(
        "/collections/default/runs/1/steps/nonexistent-model",
        params={"certain": certain},
    )

    assert response.status_code == 404
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxStepNotFoundError"
