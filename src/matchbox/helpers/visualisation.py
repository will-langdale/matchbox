import rustworkx as rx
from matplotlib.figure import Figure
from rustworkx.visualization import mpl_draw
from sqlalchemy import Engine

from matchbox.data import ENGINE
from matchbox.data.utils import get_model_subgraph


def draw_model_tree(engine: Engine = ENGINE) -> Figure:
    """
    Draws the model subgraph.
    """
    G = get_model_subgraph(engine=engine)

    node_indices = G.node_indices()
    datasets = {
        G[node_indices[i]]["id"]: i
        for i in node_indices
        if G[node_indices[i]]["type"] == "dataset"
    }

    colours = []
    for i in node_indices:
        type = G[node_indices[i]]["type"]
        if type == "dataset":
            colours.append((0, 0, 1, 0.2))
        elif type == "model":
            colours.append((1, 0, 0, 0.2))

    return mpl_draw(
        G,
        pos=rx.spring_layout(
            G,
            pos={v: [0, i / 2] for v, i in enumerate(datasets.values())},
            fixed=set(datasets.values()),
        ),
        node_color=colours,
        with_labels=True,
        labels=lambda node: node["name"],
        edge_labels=lambda edge: edge["type"],
        font_size=8,
    )
