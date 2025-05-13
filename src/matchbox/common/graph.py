"""Common data structures for resolution graphs."""

from enum import StrEnum

import rustworkx as rx
from pydantic import BaseModel, ConfigDict

from matchbox.common.dtos import ResolutionName

DEFAULT_RESOLUTION: ResolutionName = "__DEFAULT__"


class ResolutionNodeType(StrEnum):
    """Types of nodes in a resolution."""

    SOURCE = "source"
    MODEL = "model"
    HUMAN = "human"


class ResolutionNode(BaseModel):
    """A node in a resolution graph."""

    model_config = ConfigDict(frozen=True)

    id: int
    name: ResolutionName
    type: ResolutionNodeType


class ResolutionEdge(BaseModel):
    """An edge in a resolution graph."""

    model_config = ConfigDict(frozen=True)

    parent: int
    child: int


class ResolutionGraph(BaseModel):
    """A directed graph of resolution nodes and edges."""

    nodes: set[ResolutionNode]
    edges: set[ResolutionEdge]

    def to_rx(self) -> rx.PyDiGraph:
        """Convert the resolution graph to a rustworkx directed graph."""
        nodes = {}
        G = rx.PyDiGraph()
        for n in self.nodes:
            node_data = {
                "id": n.id,
                "name": n.name,
                "type": str(n.type),
            }
            nodes[n.id] = G.add_node(node_data)
        for e in self.edges:
            G.add_edge(nodes[e.parent], nodes[e.child], {})
        return G
