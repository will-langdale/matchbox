from io import BytesIO
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from matchbox.common.arrow import SCHEMA_JUDGEMENTS
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import MatchboxDataNotFound, MatchboxUserNotFoundError
from matchbox.server.api.dependencies import backend
from matchbox.server.api.main import app


def test_insert_judgement_ok(test_client: TestClient):
    """Test that a judgement is passed on to backend."""
    mock_backend = Mock()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend
    judgement = Judgement(user_id=1, clusters=[[1]])
    response = test_client.post("/eval/", json=judgement.model_dump())
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
        "/eval/", json=Judgement(user_id=1, clusters=[[1]]).model_dump()
    )
    assert response.status_code == 422
    assert response.json()["entity"] == "clusters"

    mock_backend.insert_judgement = Mock(side_effect=MatchboxUserNotFoundError)
    response = test_client.post(
        "/eval/", json=Judgement(user_id=1, clusters=[[1]]).model_dump()
    )
    assert response.status_code == 422
    assert response.json()["entity"] == "users"


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
    response = test_client.get("/eval/")
    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)

    # Check response
    assert response.status_code == 200
    assert table.equals(judgements)
