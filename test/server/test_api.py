from fastapi.testclient import TestClient
from matchbox.server import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}
