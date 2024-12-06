from enum import StrEnum

import rustworkx as rx
from matchbox.common.hash import hash_to_str
from pydantic import BaseModel


class ResolutionNodeKind(StrEnum):
    DATASET = "dataset"
    MODEL = "model"
    HUMAN = "human"


class ResolutionNode(BaseModel):
    hash: bytes
    name: str
    kind = ResolutionNodeKind

    def __hash__(self):
        return hash(self.hash)


class ResolutionEdge(BaseModel):
    parent: bytes
    child: bytes

    def __hash__(self):
        return hash((self.parent, self.child))


class ResolutionGraph(BaseModel):
    nodes: set[ResolutionNode]
    edges: set[ResolutionEdge]

    @classmethod
    def from_rx(cls):
        pass

    def to_rx(self) -> rx.PyDiGraph:
        G = rx.PyDiGraph()
        for n in self.nodes:
            G.add_node({"id": hash_to_str(n.hash), "name": n.name, "kind": str(n.kind)})
        for e in self.edges:
            G.add_edge(e.parent, e.child)
        return G
