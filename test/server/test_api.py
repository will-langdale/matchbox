from fastapi.testclient import TestClient
from matchbox.server import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_count_backend_items():
    response = client.get("/testing/count")
    assert response.status_code == 200


def test_clear_backend():
    response = client.post("/testing/clear")
    assert response.status_code == 200


def test_list_sources():
    response = client.get("/sources")
    assert response.status_code == 200


def test_get_source():
    response = client.get("/sources/test_source")
    assert response.status_code == 200


def test_insert_source():
    response = client.post("/sources")
    assert response.status_code == 200


def test_list_models():
    response = client.get("/models")
    assert response.status_code == 200


def test_get_model():
    response = client.get("/models/test_model")
    assert response.status_code == 200


def test_add_model():
    response = client.post("/models")
    assert response.status_code == 200


def test_delete_model():
    response = client.delete("/models/test_model")
    assert response.status_code == 200


def test_get_results():
    response = client.get("/models/test_model/results")
    assert response.status_code == 200


def test_set_results():
    response = client.post("/models/test_model/results")
    assert response.status_code == 200


def test_get_truth():
    response = client.get("/models/test_model/truth")
    assert response.status_code == 200


def test_set_truth():
    response = client.post("/models/test_model/truth")
    assert response.status_code == 200


def test_get_ancestors():
    response = client.get("/models/test_model/ancestors")
    assert response.status_code == 200


def test_get_ancestors_cache():
    response = client.get("/models/test_model/ancestors_cache")
    assert response.status_code == 200


def test_set_ancestors_cache():
    response = client.post("/models/test_model/ancestors_cache")
    assert response.status_code == 200


def test_query():
    response = client.get("/query")
    assert response.status_code == 200


def test_validate_hashes():
    response = client.get("/validate/hash")
    assert response.status_code == 200


def test_get_model_subgraph():
    response = client.get("/report/models")
    assert response.status_code == 200
