from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
    ResolutionNodeKind,
)
from matchbox.server import app

client = TestClient(app)


class TestMatchboxAPI:
    def test_healthcheck(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

    @patch("matchbox.server.base.BackendManager.get_backend")
    def test_count_all_backend_items(self, get_backend):
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
    def test_count_backend_item(self, get_backend):
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

    # def test_get_source():
    #     response = client.get("/sources/test_source")
    #     assert response.status_code == 200

    # def test_add_source():
    #     response = client.post("/sources")
    #     assert response.status_code == 200

    # def test_list_models():
    #     response = client.get("/models")
    #     assert response.status_code == 200

    # def test_get_model():
    #     response = client.get("/models/test_model")
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

    # def test_validate_hashes():
    #     response = client.get("/validate/hash")
    #     assert response.status_code == 200

    @patch("matchbox.server.base.BackendManager.get_backend")
    def test_get_resolution_graph(self, get_backend):
        res_graph = ResolutionGraph(
            nodes={
                ResolutionNode(
                    hash=bytes(1), name="1", kind=ResolutionNodeKind.DATASET
                ),
                ResolutionNode(
                    hash=bytes(2), name="2", kind=ResolutionNodeKind.DATASET
                ),
                ResolutionNode(hash=bytes(3), name="3", kind=ResolutionNodeKind.MODEL),
                ResolutionNode(hash=bytes(4), name="4", kind=ResolutionNodeKind.MODEL),
                ResolutionNode(hash=bytes(5), name="5", kind=ResolutionNodeKind.MODEL),
            },
            edges={
                ResolutionEdge(parent=bytes(2), child=bytes(1)),
                ResolutionEdge(parent=bytes(4), child=bytes(3)),
                ResolutionEdge(parent=bytes(5), child=bytes(2)),
                ResolutionEdge(parent=bytes(5), child=bytes(4)),
            },
        )
        mock_backend = Mock()
        mock_backend.get_resolution_graph = Mock(return_value=res_graph)
        get_backend.return_value = mock_backend

        response = client.get("/report/resolutions")
        assert response.status_code == 200
        assert ResolutionGraph.model_validate(response.json())
