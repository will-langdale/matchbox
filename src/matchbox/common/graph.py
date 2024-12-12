from enum import StrEnum

import rustworkx as rx
from matchbox.common.hash import hash_to_base64
from pydantic import BaseModel


class ResolutionNodeType(StrEnum):
    DATASET = "dataset"
    MODEL = "model"
    HUMAN = "human"


class ResolutionNode(BaseModel):
    hash: bytes
    name: str
    type: ResolutionNodeType

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

    def to_rx(self) -> rx.PyDiGraph:
        nodes = {}
        G = rx.PyDiGraph()
        for n in self.nodes:
            node_data = {
                "id": hash_to_base64(n.hash),
                "name": n.name,
                "type": str(n.type),
            }
            nodes[n.hash] = G.add_node(node_data)
        for e in self.edges:
            G.add_edge(nodes[e.parent], nodes[e.child], {})
        return G
