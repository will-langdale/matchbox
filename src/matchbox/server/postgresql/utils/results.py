from typing import NamedTuple

import pandas as pd
import pyarrow as pa
import rustworkx as rx
from sqlalchemy import Engine, and_, case, exists, func, select
from sqlalchemy.orm import Session

from matchbox.common.results import (
    ClusterResults,
    ModelMetadata,
    ModelType,
    ProbabilityResults,
)
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Models,
    ModelsFrom,
    Probabilities,
)
from matchbox.server.postgresql.utils.query import hash_to_hex_decode


class SourceInfo(NamedTuple):
    """Information about a model's sources."""

    left: bytes
    right: bytes | None
    left_ancestors: set[bytes]
    right_ancestors: set[bytes] | None


def _get_model_parents(engine: Engine, model_hash: bytes) -> tuple[bytes, bytes | None]:
    """Get the model's immediate parent models."""
    parent_query = (
        select(Models.hash, Models.type)
        .join(ModelsFrom, Models.hash == ModelsFrom.parent)
        .where(ModelsFrom.child == model_hash)
        .where(ModelsFrom.level == 1)
    )

    with engine.connect() as conn:
        parents = conn.execute(parent_query).fetchall()

    if len(parents) == 1:
        return parents[0][0], None
    elif len(parents) == 2:
        p1, p2 = parents
        p1_hash, p1_type = p1
        p2_hash, p2_type = p2
        # Put dataset first if it exists
        if p1_type == "dataset":
            return p1_hash, p2_hash
        elif p2_type == "dataset":
            return p2_hash, p1_hash
        # Both models, maintain original order
        return p1_hash, p2_hash
    else:
        raise ValueError(f"Model has unexpected number of parents: {len(parents)}")


def _get_source_info(engine: Engine, model_hash: bytes) -> SourceInfo:
    """Get source models and their ancestry information."""
    left_hash, right_hash = _get_model_parents(engine=engine, model_hash=model_hash)

    with Session(engine) as session:
        left = session.get(Models, left_hash)
        right = session.get(Models, right_hash) if right_hash else None

        left_ancestors = {left_hash} | {m.hash for m in left.ancestors}
        if right:
            right_ancestors = {right_hash} | {m.hash for m in right.ancestors}
        else:
            right_ancestors = None

    return SourceInfo(
        left=left_hash,
        right=right_hash,
        left_ancestors=left_ancestors,
        right_ancestors=right_ancestors,
    )


def _get_leaf_pair_clusters(engine: Engine, model_hash: bytes) -> list[tuple]:
    """Get all clusters with exactly two leaf children."""
    # Subquery to identify leaf nodes
    leaf_nodes = ~exists().where(Contains.parent == Clusters.hash).correlate(Clusters)

    query = (
        select(
            Contains.parent.label("parent_hash"),
            Probabilities.probability,
            func.array_agg(Clusters.hash).label("child_hashes"),
            func.array_agg(Clusters.dataset).label("child_datasets"),
            func.array_agg(Clusters.id).label("child_ids"),
        )
        .join(
            Probabilities,
            and_(
                Probabilities.cluster == Contains.parent,
                Probabilities.model == model_hash,
            ),
        )
        .join(Clusters, Clusters.hash == Contains.child)
        .where(leaf_nodes)
        .group_by(Contains.parent, Probabilities.probability)
        .having(func.count() == 2)
    )

    with engine.connect() as conn:
        return conn.execute(query).fetchall()


def _determine_hash_order(
    engine: Engine,
    hashes: list[bytes],
    datasets: list[bytes],
    left_source: Models,
    left_ancestors: set[bytes],
) -> tuple[int, int]:
    """Determine which child corresponds to left/right source."""
    # Check dataset assignment first
    if datasets[0] == left_source.hash:
        return 0, 1
    elif datasets[1] == left_source.hash:
        return 1, 0

    # Check probability ancestry
    left_prob_query = (
        select(Probabilities)
        .where(Probabilities.cluster == hashes[0])
        .where(Probabilities.model.in_(left_ancestors))
    )
    with engine.connect() as conn:
        has_left_prob = conn.execute(left_prob_query).fetchone() is not None

    return (0, 1) if has_left_prob else (1, 0)


def _get_immediate_children(graph: rx.PyDiGraph, node_id: int) -> set[int]:
    """Get immediate child node IDs of a given node in the graph."""
    return {edge[1] for edge in graph.out_edges(node_id)}


def _is_leaf(graph: rx.PyDiGraph, node_id: int) -> bool:
    """Check if a node is a leaf (has no children)."""
    return len(graph.out_edges(node_id)) == 0


def get_model_probabilities(engine: Engine, model: Models) -> ProbabilityResults:
    """
    Recover the model's ProbabilityResults.

    For each probability this model assigned:
    - Get its two immediate children
    - Filter for children that aren't parents of other clusters this model scored
    - Determine left/right by tracing ancestry to source models using query helpers

    Args:
        engine: SQLAlchemy engine
        model: Model instance to query

    Returns:
        ProbabilityResults containing the original pairwise probabilities
    """
    source_info: SourceInfo = _get_source_info(engine=engine, model_hash=model.hash)

    with Session(engine) as session:
        left = session.get(Models, source_info.left)
        right = session.get(Models, source_info.right) if source_info.right else None

        metadata = ModelMetadata(
            name=model.name,
            description=model.description or "",
            type=ModelType.DEDUPER if source_info.right is None else ModelType.LINKER,
            left_source=left.name,
            right_source=right.name if source_info.right else None,
        )

        # First get all clusters this model assigned probabilities to
        model_clusters = (
            select(Probabilities.cluster)
            .where(Probabilities.model == hash_to_hex_decode(model.hash))
            .cte("model_clusters")
        )

        # Get clusters that are parents in Contains for model's probabilities
        model_parents = (
            select(Contains.parent)
            .join(model_clusters, Contains.child == model_clusters.c.cluster)
            .cte("model_parents")
        )

        # Get valid pairs (those with exactly 2 children)
        # where neither child is a parent in the model's hierarchy
        valid_pairs = (
            select(Contains.parent)
            .join(
                Probabilities,
                and_(
                    Probabilities.cluster == Contains.parent,
                    Probabilities.model == hash_to_hex_decode(model.hash),
                ),
            )
            .where(~Contains.child.in_(select(model_parents)))
            .group_by(Contains.parent)
            .having(func.count() == 2)
            .cte("valid_pairs")
        )

        # Join to get children and probabilities
        pairs = (
            select(
                Contains.parent.label("hash"),
                func.array_agg(
                    case(
                        (
                            Contains.child.in_(list(source_info.left_ancestors)),
                            Contains.child,
                        ),
                        (
                            Contains.child.in_(list(source_info.right_ancestors))
                            if source_info.right_ancestors
                            else Contains.child.notin_(
                                list(source_info.left_ancestors)
                            ),
                            Contains.child,
                        ),
                    )
                ).label("children"),
                func.min(Probabilities.probability).label("probability"),
            )
            .join(valid_pairs, valid_pairs.c.parent == Contains.parent)
            .join(
                Probabilities,
                and_(
                    Probabilities.cluster == Contains.parent,
                    Probabilities.model == hash_to_hex_decode(model.hash),
                ),
            )
            .group_by(Contains.parent)
        ).cte("pairs")

        # Final select to properly split out left and right
        final_select = select(
            pairs.c.hash,
            pairs.c.children[1].label("left_id"),
            pairs.c.children[2].label("right_id"),
            pairs.c.probability,
        )

        results = session.execute(final_select).fetchall()

        df = pd.DataFrame(
            results, columns=["hash", "left_id", "right_id", "probability"]
        ).astype(
            {
                "hash": pd.ArrowDtype(pa.binary()),
                "left_id": pd.ArrowDtype(pa.binary()),
                "right_id": pd.ArrowDtype(pa.binary()),
                "probability": pd.ArrowDtype(pa.float32()),
            }
        )

        return ProbabilityResults(dataframe=df, metadata=metadata)


def _get_all_leaf_descendants(graph: rx.PyDiGraph, node_id: int) -> set[int]:
    """Get all leaf descendant node IDs of a given node in the graph."""
    descendants = set()
    to_process = [node_id]

    while to_process:
        current = to_process.pop()
        children = [edge[1] for edge in graph.out_edges(current)]

        if not children:
            descendants.add(current)
        else:
            to_process.extend(children)

    return descendants


def get_model_clusters(engine: Engine, model: Models) -> ClusterResults:
    """
    Recover the model's Clusters.

    Clusters are the connected components of the model at every threshold.

    While they're stored in a hierarchical structure, we need to recover the
    original components, where all child hashes are leaf Clusters.

    Args:
        engine: SQLAlchemy engine
        model: Model instance to query

    Returns:
        A ClusterResults object containing connected components and model metadata
    """
    source_info: SourceInfo = _get_source_info(engine=engine, model_hash=model.hash)

    with Session(engine) as session:
        # Build metadata
        left = session.get(Models, source_info.left)
        right = session.get(Models, source_info.right) if source_info.right else None

        metadata = ModelMetadata(
            name=model.name,
            description=model.description or "",
            type=ModelType.DEDUPER if source_info.right is None else ModelType.LINKER,
            left_source=left.name,
            right_source=right.name if source_info.right else None,
        )

        # Get all clusters and their relationships for this model
        hierarchy_query = (
            select(Contains.parent, Contains.child, Probabilities.probability)
            .join(
                Probabilities,
                and_(
                    Probabilities.cluster == Contains.parent,
                    Probabilities.model == model.hash,
                ),
            )
            .order_by(Probabilities.probability.desc())
        )

        hierarchy = session.execute(hierarchy_query).fetchall()

        # Get all leaf nodes (clusters with no children) and their IDs
        leaf_query = select(Clusters.hash, Clusters.id).where(
            ~Clusters.hash.in_(select(Contains.parent).distinct())
        )
        leaf_nodes = {
            row.hash: row.id[0] if row.id else None
            for row in session.execute(leaf_query)
        }

        # Get unique thresholds and components at each threshold
        threshold_query = (
            select(Probabilities.cluster, Probabilities.probability)
            .where(Probabilities.model == model.hash)
            .order_by(Probabilities.probability.desc())
        )
        threshold_components = session.execute(threshold_query).fetchall()

    # Build directed graph of the full hierarchy
    graph = rx.PyDiGraph()
    nodes: dict[bytes, int] = {}  # node_hash -> node_id

    def get_node_id(hash: bytes) -> int:
        if hash not in nodes:
            nodes[hash] = graph.add_node(hash)
        return nodes[hash]

    for parent, child, prob in hierarchy:
        parent_id = get_node_id(parent)
        child_id = get_node_id(child)
        graph.add_edge(parent_id, child_id, prob)

    # Process each threshold level
    components: list[tuple[bytes, bytes, float]] = []
    seen_combinations = set()

    threshold_groups = {}
    for comp, thresh in threshold_components:
        if thresh not in threshold_groups:
            threshold_groups[thresh] = []
        threshold_groups[thresh].append(comp)

    # Process thresholds in descending order
    for threshold in sorted(threshold_groups.keys(), reverse=True):
        for component in threshold_groups[threshold]:
            component_id = get_node_id(component)

            leaf_ids = _get_all_leaf_descendants(graph, component_id)

            leaf_hashes = {
                graph.get_node_data(leaf_id)
                for leaf_id in leaf_ids
                if graph.get_node_data(leaf_id) in leaf_nodes
            }

            for leaf in leaf_hashes:
                if leaf_nodes[leaf] is not None:
                    relation = (component, leaf, threshold)
                    if relation not in seen_combinations:
                        components.append(relation)
                        seen_combinations.add(relation)

    df = pd.DataFrame(components, columns=["parent", "child", "threshold"]).astype(
        {
            "parent": pd.ArrowDtype(pa.binary()),
            "child": pd.ArrowDtype(pa.binary()),
            "threshold": pd.ArrowDtype(pa.float32()),
        }
    )

    return ClusterResults(
        dataframe=df,
        metadata=metadata,
    )
