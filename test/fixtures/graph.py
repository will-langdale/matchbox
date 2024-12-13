import pytest
from rustworkx import PyDiGraph

from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
)
from matchbox.common.graph import (
    ResolutionNodeType as ResType,
)
from matchbox.common.hash import hash_to_base64


@pytest.fixture
def resolution_graph() -> ResolutionGraph:
    res_graph = ResolutionGraph(
        nodes={
            ResolutionNode(hash=bytes(1), name="1", type=ResType.DATASET),
            ResolutionNode(hash=bytes(2), name="2", type=ResType.DATASET),
            ResolutionNode(hash=bytes(3), name="3", type=ResType.MODEL),
            ResolutionNode(hash=bytes(4), name="4", type=ResType.MODEL),
            ResolutionNode(hash=bytes(5), name="5", type=ResType.MODEL),
        },
        edges={
            ResolutionEdge(parent=bytes(2), child=bytes(1)),
            ResolutionEdge(parent=bytes(4), child=bytes(3)),
            ResolutionEdge(parent=bytes(5), child=bytes(2)),
            ResolutionEdge(parent=bytes(5), child=bytes(4)),
        },
    )

    return res_graph


@pytest.fixture
def pydigraph() -> PyDiGraph:
    def make_id(n: int) -> str:
        return hash_to_base64(bytes(n))

    G = PyDiGraph()
    n1 = G.add_node({"id": make_id(1), "name": "1", "type": str(ResType.DATASET)})
    n2 = G.add_node({"id": make_id(2), "name": "2", "type": str(ResType.DATASET)})
    n3 = G.add_node({"id": make_id(3), "name": "3", "type": str(ResType.MODEL)})
    n4 = G.add_node({"id": make_id(4), "name": "4", "type": str(ResType.MODEL)})
    n5 = G.add_node({"id": make_id(5), "name": "5", "type": str(ResType.MODEL)})
    G.add_edge(n2, n1, {})
    G.add_edge(n4, n3, {})
    G.add_edge(n5, n2, {})
    G.add_edge(n5, n4, {})
    return G
