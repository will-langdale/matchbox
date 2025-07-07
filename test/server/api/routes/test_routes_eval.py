from io import BytesIO
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from matchbox.common.arrow import SCHEMA_EVAL_SAMPLES, SCHEMA_JUDGEMENTS
from matchbox.common.dtos import BackendUnprocessableType
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxResolutionNotFoundError,
    MatchboxTooManySamplesRequested,
    MatchboxUserNotFoundError,
)
from matchbox.server.api.dependencies import backend
from matchbox.server.api.main import app


def test_insert_judgement_ok(test_client: TestClient):
    """Test that a judgement is passed on to backend."""
    mock_backend = Mock()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend
    judgement = Judgement(user_id=1, clusters=[[1]])
    response = test_client.post("/eval/judgements", json=judgement.model_dump())
    assert response.status_code == 201
    assert (
        mock_backend.insert_judgement.call_args_list[0].kwargs["judgement"] == judgement
    )


def test_insert_judgement_error(test_client: TestClient):
    mock_backend = Mock()
    mock_backend.insert_judgement = Mock(side_effect=MatchboxDataNotFound)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.post(
        "/eval/judgements", json=Judgement(user_id=1, clusters=[[1]]).model_dump()
    )
    assert response.status_code == 422
    assert response.json()["entity"] == BackendUnprocessableType.CLUSTER

    mock_backend.insert_judgement = Mock(side_effect=MatchboxUserNotFoundError)
    response = test_client.post(
        "/eval/judgements", json=Judgement(user_id=1, clusters=[[1]]).model_dump()
    )
    assert response.status_code == 422
    assert response.json()["entity"] == BackendUnprocessableType.USER


def test_get_judgements(test_client: TestClient):
    """Test that all judgements can be retrieved."""
    judgements = pa.Table.from_pylist(
        [
            {"user_id": 1, "parent": 1, "child": 2},
            {"user_id": 1, "parent": 1, "child": 3},
        ],
        schema=SCHEMA_JUDGEMENTS,
    )

    mock_backend = Mock()
    mock_backend.get_judgements = Mock(return_value=judgements)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Process response
    response = test_client.get("/eval/judgements")
    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)

    # Check response
    assert response.status_code == 200
    assert table.equals(judgements)


def test_get_samples(test_client: TestClient):
    """Test that samples can be requested."""
    sample = pa.Table.from_pylist(
        [
            {"mb_id": 1, "key": "1", "source": "source_a"},
            {"mb_id": 1, "key": "2", "source": "source_a"},
        ],
        schema=SCHEMA_EVAL_SAMPLES,
    )

    mock_backend = Mock()
    mock_backend.sample_for_eval = Mock(return_value=sample)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Process response
    response = test_client.get(
        "/eval/samples",
        params={"n": 10, "resolution": "resolution", "user_id": 12},
    )
    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)

    # Check response
    assert response.status_code == 200
    assert table.equals(sample)


@pytest.mark.parametrize(
    ("exception", "entity"),
    [
        pytest.param(
            MatchboxUserNotFoundError,
            BackendUnprocessableType.USER,
            id="user_not_found",
        ),
        pytest.param(
            MatchboxResolutionNotFoundError,
            BackendUnprocessableType.RESOLUTION,
            id="resolution_not_found",
        ),
        pytest.param(
            MatchboxTooManySamplesRequested,
            BackendUnprocessableType.SAMPLE_SIZE,
            id="too_many_samples_requested",
        ),
    ],
)
def test_get_samples_error(exception, entity, test_client: TestClient):
    """Test errors in requesting samples."""
    mock_backend = Mock()

    mock_backend.sample_for_eval = Mock(side_effect=exception)

    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.get(
        "/eval/samples",
        params={"n": 10, "resolution": "resolution", "user_id": 12},
    )

    assert response.status_code == 422
    assert response.json()["entity"] == entity
