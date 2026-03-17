import json

import polars as pl
import pytest
from httpx import Response
from polars.testing import assert_frame_equal
from respx.router import MockRouter
from sqlalchemy import Engine

from matchbox.client.dags import DAG
from matchbox.client.models import Model, add_model_class
from matchbox.client.models.linkers.base import LinkerSettings
from matchbox.client.queries import Query
from matchbox.common.arrow import SCHEMA_MODEL_EDGES, table_to_buffer
from matchbox.common.dtos import (
    CRUDOperation,
    ErrorResponse,
    ModelConfig,
    ModelType,
    ResourceOperationStatus,
    Step,
    UploadInfo,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxServerFileError,
)
from matchbox.common.factories.models import MockLinker, model_factory
from matchbox.common.factories.sources import source_factory


def test_init_and_run_model(
    sqla_sqlite_warehouse: Engine, matchbox_api: MockRouter
) -> None:
    """Test that model can be initialised and run correctly."""
    # Register "custom" model
    add_model_class(MockLinker)

    # Mock API
    foo = source_factory(engine=sqla_sqlite_warehouse).write_to_location()
    foo.source.run()
    bar = source_factory(engine=sqla_sqlite_warehouse).write_to_location()
    bar.source.run()

    # Mock API
    query_endpoint = matchbox_api.get("/query").mock(
        side_effect=[
            # First query
            Response(200, content=table_to_buffer(foo.data).read()),
            Response(200, content=table_to_buffer(bar.data).read()),
            # Second query (for pre-fetching)
            Response(200, content=table_to_buffer(foo.data).read()),
            Response(200, content=table_to_buffer(bar.data).read()),
        ]
    )
    dag = DAG("collection")
    foo_query = Query(foo.source, dag=dag)
    bar_query = Query(bar.source, dag=dag)

    model = Model(
        dag=dag,
        name="name",
        description="description",
        model_class=MockLinker,
        model_settings=LinkerSettings(),
        left_query=foo_query,
        right_query=bar_query,
    )

    assert model.config == ModelConfig(
        type=ModelType.LINKER,
        model_class="MockLinker",
        model_settings=json.dumps({"left_id": "l.field", "right_id": "r.field"}),
        left_query=foo_query.config,
        right_query=bar_query.config,
    )

    results = model.run()
    assert_frame_equal(results, model.results)
    assert results.schema == pl.Schema(SCHEMA_MODEL_EDGES)

    # Can use pre-fetched query data
    left_df, right_df = foo_query.data(), bar_query.data()
    old_query_count = query_endpoint.call_count
    model.run(left_df, right_df)
    assert query_endpoint.call_count == old_query_count


def test_model_sync(matchbox_api: MockRouter) -> None:
    # Mock model
    testkit = model_factory(model_type="linker")

    # Mock the routes:
    # Step doesn't yet exist
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}"
    ).mock(
        return_value=Response(
            404,
            json=ErrorResponse(
                exception_type="MatchboxStepNotFoundError",
                message="Model not found",
            ).model_dump(),
        )
    )

    # Step can be inserted
    insert_config_route = matchbox_api.post(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}"
    ).mock(
        return_value=Response(
            201,
            content=ResourceOperationStatus(
                success=True,
                target=f"Step {testkit.model.name}",
                operation=CRUDOperation.CREATE,
            ).model_dump_json(),
        )
    )

    # Step data can be inserted
    insert_results_route = matchbox_api.post(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}/data"
    ).mock(
        return_value=Response(
            202,
            json=ResourceOperationStatus(
                success=True, target="", operation=CRUDOperation.CREATE
            ).model_dump(),
        )
    )

    # Later, step can be updated
    update_route = matchbox_api.put(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}"
    ).mock(
        return_value=Response(
            200,
            content=ResourceOperationStatus(
                success=True,
                target=f"Step {testkit.model.name}",
                operation=CRUDOperation.UPDATE,
            ).model_dump_json(),
        )
    )

    # Later, step can be deleted and recreated
    delete_route = matchbox_api.delete(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}"
    ).mock(
        return_value=Response(
            200,
            content=ResourceOperationStatus(
                success=True,
                target=f"Step {testkit.model.name}",
                operation=CRUDOperation.DELETE,
            ).model_dump_json(),
        )
    )

    # -- ERRORS --

    # Can't sync before running
    with pytest.raises(RuntimeError, match="must be run"):
        testkit.model.sync()

    # We now run, but test that upload failure is handled
    testkit.fake_run()
    # If stage is not PROCESSING, job will fail
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}/data/status"
    ).mock(
        return_value=Response(
            200, json=UploadInfo(stage=UploadStage.READY).model_dump()
        )
    )
    with pytest.raises(MatchboxServerFileError, match="issue"):
        testkit.model.sync()

    # -- FIRST TIME INSERTION --

    # Before upload, step is ready for data, after it is complete
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}/data/status"
    ).mock(
        return_value=Response(
            200, json=UploadInfo(stage=UploadStage.COMPLETE).model_dump()
        )
    )

    # Sync the source, successfully
    testkit.model.sync()
    # Source was created, not updated or deleted
    assert insert_config_route.called
    assert not update_route.called
    assert not delete_route.called
    # Step metadata was correct
    step_call = Step.model_validate_json(
        insert_config_route.calls.last.request.content.decode("utf-8")
    )
    assert step_call == testkit.model.to_dto()
    # Step data was correct
    assert (
        b"Content-Disposition: form-data;"
        in insert_results_route.calls.last.request.content
    )
    assert b"PAR1" in insert_results_route.calls.last.request.content

    # -- SOFT UPDATE --

    insert_results_route.reset()
    # Mock endpoint now returns existing step
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}"
    ).mock(return_value=Response(200, json=testkit.model.to_dto().model_dump()))

    # Mock endpoint now declares data is present already
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}/data/status"
    ).mock(
        return_value=Response(
            200, json=UploadInfo(stage=UploadStage.COMPLETE).model_dump()
        )
    )

    testkit.model.sync()
    # Step was compatible: ensure it was updated, not deleted
    assert update_route.called
    assert not delete_route.called
    # The data did not need to be updated
    assert not insert_results_route.called

    # -- HARD UPDATE --

    insert_results_route.reset()
    # Mock endpoint now returns existing step
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}"
    ).mock(return_value=Response(200, json=testkit.model.to_dto().model_dump()))

    # Changing local model results requires deletion and re-insertion
    assert testkit.model.results is not None
    testkit.model.results = testkit.model.results.slice(1, 3)

    # Step data is first ready to upload, and then uploaded
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}/data/status"
    ).mock(
        return_value=Response(
            200, json=UploadInfo(stage=UploadStage.COMPLETE).model_dump()
        )
    )

    testkit.model.sync()
    assert delete_route.called
    assert insert_results_route.called


def test_delete_step(matchbox_api: MockRouter) -> None:
    """Test successfully deleting a step."""
    # Create test model using factory
    testkit = model_factory()

    # Mock the DELETE endpoint with success response
    route = matchbox_api.delete(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}",
        params={"certain": True},
    ).mock(
        return_value=Response(
            200,
            json=ResourceOperationStatus(
                success=True,
                target=f"Step {testkit.model.name}",
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


def test_delete_step_needs_confirmation(matchbox_api: MockRouter) -> None:
    """Test attempting to delete a step without confirmation returns 409."""
    # Create test model using factory
    testkit = model_factory()

    # Mock the DELETE endpoint with 409 confirmation required response
    route = matchbox_api.delete(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/steps/{testkit.model.name}"
    ).mock(
        return_value=Response(
            409,
            json=ErrorResponse(
                exception_type="MatchboxDeletionNotConfirmed",
                message="Cannot delete model with dependent models: dedupe1, dedupe2",
                details={"children": ["dedupe1", "dedupe2"]},
            ).model_dump(),
        )
    )

    # Attempt to delete without certain=True
    with pytest.raises(MatchboxDeletionNotConfirmed):
        testkit.model.delete()

    # Verify the response and API call
    assert route.called
    assert route.calls.last.request.url.params["certain"] == "false"
