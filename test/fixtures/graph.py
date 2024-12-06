import pytest
from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
    ResolutionNodeKind,
)


@pytest.fixture
def resolution_graph() -> ResolutionGraph:
    res_graph = ResolutionGraph(
        nodes={
            ResolutionNode(hash=bytes(1), name="1", kind=ResolutionNodeKind.DATASET),
            ResolutionNode(hash=bytes(2), name="2", kind=ResolutionNodeKind.DATASET),
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

    return res_graph
