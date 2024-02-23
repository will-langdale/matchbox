import rustworkx as rx
from matplotlib.figure import Figure
from rustworkx.visualization import mpl_draw
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cmf.data import ENGINE, Models, ModelsFrom, SourceDataset


def draw_model_tree(engine: Engine = ENGINE) -> Figure:
    """
    Draws the model subgraph.
    """
    G = rx.PyDiGraph()
    models = {}
    datasets = {}

    with Session(engine) as session:
        for dataset in session.query(SourceDataset).all():
            dataset_idx = G.add_node(
                {
                    "id": str(dataset.uuid),
                    "name": f"{dataset.db_schema}.{dataset.db_table}",
                    "type": "dataset",
                }
            )
            datasets[dataset.uuid] = dataset_idx

        for model in session.query(Models).all():
            model_idx = G.add_node(
                {"id": str(model.sha1), "name": model.name, "type": "model"}
            )
            models[model.sha1] = model_idx
            if model.deduplicates is not None:
                dataset_idx = datasets.get(model.deduplicates)
                _ = G.add_edge(model_idx, dataset_idx, {"type": "deduplicates"})

        for edge in session.query(ModelsFrom).all():
            parent_idx = models.get(edge.parent)
            child_idx = models.get(edge.child)
            _ = G.add_edge(parent_idx, child_idx, {"type": "from"})

    node_indices = G.node_indices()

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
