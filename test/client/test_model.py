import json
from datetime import datetime
from unittest.mock import Mock, patch

import polars as pl
import pyarrow as pa
import pytest
from httpx import Response
from respx.router import MockRouter

from matchbox.client.models import Model
from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.client.queries import Query
from matchbox.client.results import Results
from matchbox.common.arrow import SCHEMA_QUERY_WITH_LEAVES, SCHEMA_RESULTS
from matchbox.common.dtos import (
    BackendResourceType,
    BackendUploadType,
    CRUDOperation,
    ModelConfig,
    ModelType,
    NotFoundError,
    ResolutionOperationStatus,
    UploadStage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxServerFileError,
    MatchboxUnhandledServerResponse,
)
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import source_factory


class MockLinker(Linker):
    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        return self

    def link(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        return pl.from_arrow(pa.Table.from_pylist([], schema=SCHEMA_RESULTS))


@patch("matchbox.client.models.models.Query.run")
def test_init_and_run_model(mock_run: Mock):
    """Test that model can be initialised and run correctly."""
    foo_query = Query(source_factory().source)
    bar_query = Query(source_factory().source)

    mock_run.return_value = pl.from_arrow(
        pa.Table.from_pylist(
            [{"id": 1, "key": "a", "leaf_id": 1}],
            schema=SCHEMA_QUERY_WITH_LEAVES,
        )
    )

    model = Model(
        name="name",
        description="description",
        model_class=MockLinker,
        model_settings=LinkerSettings(left_id="left", right_id="right"),
        query=foo_query,
        right_query=bar_query,
    )

    model.run()

    assert model.config == ModelConfig(
        type=ModelType.LINKER,
        model_class="MockLinker",
        model_settings=json.dumps({"left_id": "left", "right_id": "right"}),
        query=foo_query.config,
        right_query=bar_query.config,
    )

    assert model.results.left_data is None
    assert model.results.right_data is None

    model.for_validation = True
    model.run()
    assert model.results.left_data is not None
    assert model.results.right_data is not None


def test_insert_model(matchbox_api: MockRouter):
    """Test inserting a model via the API."""
    # Create test model using factory
    testkit = model_factory(model_type="linker")

    # Mock the POST /resolutions endpoint
    get_route = matchbox_api.get(f"/resolutions/{testkit.model.name}").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Model not found", entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        )
    )
    insert_route = matchbox_api.post("/resolutions").mock(
        return_value=Response(
            201,
            json=ResolutionOperationStatus(
                success=True,
                name=testkit.model.name,
                operation=CRUDOperation.CREATE,
            ).model_dump(),
        )
    )

    # Call insert_model
    testkit.model.insert_model()

    # Verify the API call
    assert get_route.called
    assert insert_route.called
    assert (
        insert_route.calls.last.request.content.decode()
        == testkit.model.to_resolution().model_dump_json()
    )


def test_insert_model_error(matchbox_api: MockRouter):
    """Test handling of model insertion errors."""
    testkit = model_factory(model_type="linker")

    # Mock the POST /resolutions endpoint with an error response
    get_route = matchbox_api.get(f"/resolutions/{testkit.model.name}").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Model not found", entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        )
    )
    insert_route = matchbox_api.post("/resolutions").mock(
        return_value=Response(
            500,
            json=ResolutionOperationStatus(
                success=False,
                name=testkit.model.name,
                operation=CRUDOperation.CREATE,
                details="Internal server error",
            ).model_dump(),
        )
    )

    # Call insert_model and verify it raises an exception
    with pytest.raises(MatchboxUnhandledServerResponse, match="Internal server error"):
        testkit.model.insert_model()

    assert get_route.called
    assert insert_route.called


def test_insert_results(matchbox_api: MockRouter):
    """Test setting model results via the API."""
    testkit = model_factory(model_type="linker")

    # Mock the endpoints needed for results upload
    init_route = matchbox_api.post(f"/resolutions/{testkit.model.name}/results").mock(
        return_value=Response(
            202,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.AWAITING_UPLOAD,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.RESULTS,
            ).model_dump_json(),
        )
    )

    upload_route = matchbox_api.post("/upload/test-upload-id").mock(
        return_value=Response(
            202,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.PROCESSING,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.RESULTS,
            ).model_dump_json(),
        )
    )

    status_route = matchbox_api.get("/upload/test-upload-id/status").mock(
        return_value=Response(
            200,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.COMPLETE,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.RESULTS,
            ).model_dump_json(),
        )
    )

    # Set results
    test_results = Results(
        probabilities=testkit.probabilities, metadata=testkit.model.config
    )
    testkit.model.results = test_results

    # Verify API calls
    assert init_route.called
    assert upload_route.called
    assert status_route.called
    assert (
        b"PAR1" in upload_route.calls.last.request.content
    )  # Check for parquet file signature


def test_results_setter_upload_failure(matchbox_api: MockRouter):
    """Test handling of upload failures when setting results."""
    testkit = model_factory(model_type="linker")

    # Mock the initial POST endpoint
    init_route = matchbox_api.post(f"/resolutions/{testkit.model.name}/results").mock(
        return_value=Response(
            202,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.AWAITING_UPLOAD,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.RESULTS,
            ).model_dump_json(),
        )
    )

    # Mock the upload endpoint with a failure
    upload_route = matchbox_api.post("/upload/test-upload-id").mock(
        return_value=Response(
            400,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.FAILED,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.RESULTS,
                details="Invalid data format",
            ).model_dump_json(),
        )
    )

    # Attempt to set results and verify it raises an exception
    test_results = Results(
        probabilities=testkit.probabilities, metadata=testkit.model.config
    )
    with pytest.raises(MatchboxServerFileError, match="Invalid data format"):
        testkit.model.results = test_results

    assert init_route.called
    assert upload_route.called


def test_truth_getter(matchbox_api: MockRouter):
    """Test getting model truth threshold from config."""
    # Create testkit with specific truth value
    testkit = model_factory(model_type="linker")
    # Update the model to have a truth value
    testkit.model._truth = 90  # Integer truth value (90 = 0.9 as float)

    # Get truth as float
    truth = testkit.model.truth

    # Verify it returns the correct value converted to float
    assert truth == 0.9


def test_truth_setter(matchbox_api: MockRouter):
    """Test setting model truth threshold via the API."""
    testkit = model_factory(model_type="linker")

    # Mock the PATCH /resolutions/{name}/truth endpoint
    route = matchbox_api.patch(f"/resolutions/{testkit.model.name}/truth").mock(
        return_value=Response(
            200,
            json=ResolutionOperationStatus(
                success=True,
                name=testkit.model.name,
                operation=CRUDOperation.UPDATE,
            ).model_dump(),
        )
    )

    # Set truth using the setter that triggers API call
    testkit.model.truth = 0.9

    # Verify the API call
    assert route.called
    assert float(route.calls.last.request.read()) == 90


def test_truth_setter_validation_error(matchbox_api: MockRouter):
    """Test setting invalid truth values."""
    testkit = model_factory(model_type="linker")

    # Attempt to set an invalid truth value using the validated setter
    with pytest.raises(ValueError):
        testkit.model.truth = 1.5


def test_delete_resolution(matchbox_api: MockRouter):
    """Test successfully deleting a resolution."""
    # Create test model using factory
    testkit = model_factory()

    # Mock the DELETE endpoint with success response
    route = matchbox_api.delete(
        f"/resolutions/{testkit.model.name}", params={"certain": True}
    ).mock(
        return_value=Response(
            200,
            json=ResolutionOperationStatus(
                success=True,
                name=testkit.model.name,
                operation=CRUDOperation.DELETE,
            ).model_dump(),
        )
    )

    # Delete the model
    response = testkit.model.delete(certain=True)

    # Verify the response and API call
    assert response
    assert route.called
    assert route.calls.last.request.url.params["certain"] == "true"


def test_delete_resolution_needs_confirmation(matchbox_api: MockRouter):
    """Test attempting to delete a resolution without confirmation returns 409."""
    # Create test model using factory
    testkit = model_factory()

    # Mock the DELETE endpoint with 409 confirmation required response
    error_details = "Cannot delete model with dependent models: dedupe1, dedupe2"
    route = matchbox_api.delete(f"/resolutions/{testkit.model.name}").mock(
        return_value=Response(
            409,
            json=ResolutionOperationStatus(
                success=False,
                name=testkit.model.name,
                operation=CRUDOperation.DELETE,
                details=error_details,
            ).model_dump(),
        )
    )

    # Attempt to delete without certain=True
    with pytest.raises(MatchboxDeletionNotConfirmed):
        testkit.model.delete()

    # Verify the response and API call
    assert route.called
    assert route.calls.last.request.url.params["certain"] == "false"
