"""Visualisation utilities."""

import rustworkx as rx
from matplotlib.figure import Figure
from rustworkx.visualization import mpl_draw

from matchbox.client._handler import get_resolution_graph
from matchbox.common.graph import ResolutionNodeType


def draw_resolution_graph() -> Figure:
    """Draws the resolution graph."""
    G: rx.PyDiGraph = get_resolution_graph().to_rx()

    node_indices = G.node_indices()
    sources = {
        G[node_indices[i]]["id"]: i
        for i in node_indices
        if G[node_indices[i]]["type"] == ResolutionNodeType.SOURCE.value
    }

    colours = []
    for i in node_indices:
        node_type = G[node_indices[i]]["type"]
        if node_type == ResolutionNodeType.SOURCE.value:
            colours.append((0, 0, 1, 0.2))
        elif node_type == ResolutionNodeType.MODEL.value:
            colours.append((1, 0, 0, 0.2))

    return mpl_draw(
        G,
        pos=rx.spring_layout(
            G,
            pos={v: [0, i / 2] for v, i in enumerate(sources.values())},
            fixed=set(sources.values()),
        ),
        node_color=colours,
        with_labels=True,
        labels=lambda node: node["name"],
        font_size=8,
    )
