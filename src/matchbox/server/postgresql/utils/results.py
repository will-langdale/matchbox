from typing import NamedTuple

import pandas as pd
import pyarrow as pa
from sqlalchemy import Engine, and_, exists, func, literal, select
from sqlalchemy.orm import Session

from matchbox.common.db import sql_to_df
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

        left_ancestors = {left_hash} | {hash for hash in left.ancestors}
        if right:
            right_ancestors = {right_hash} | {hash for hash in right.ancestors}
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


def get_model_probabilities(engine: Engine, model: Models) -> ProbabilityResults:
    """
    Recover the model's ProbabilityResults.

    Probabilities are the model's Clusters identified by:

    * Exactly two children
    * Both children are leaf nodes (not parents in Contains table)

    Args:
        engine: SQLAlchemy engine
        model: Model instance to query

    Returns:
        A ProbabilityResults object containing pairwise probabilities and model metadata
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

    results = _get_leaf_pair_clusters(engine=engine, model_hash=model.hash)

    # Process results into pairs
    rows: dict[str, list] = {
        "hash": [],
        "left_id": [],
        "right_id": [],
        "probability": [],
    }
    for parent_hash, prob, child_hashes, child_datasets, child_ids in results:
        if metadata.type == ModelType.LINKER:
            left_idx, right_idx = _determine_hash_order(
                engine=engine,
                hashes=child_hashes,
                datasets=child_datasets,
                left_source=source_info.left,
                left_ancestors=source_info.left_ancestors,
            )
        else:
            # For dedupers, order doesn't matter
            left_idx, right_idx = 0, 1

        rows["hash"].append(parent_hash)
        rows["left_id"].append(child_ids[left_idx][0])
        rows["right_id"].append(child_ids[right_idx][0])
        rows["probability"].append(prob)

    return ProbabilityResults(
        dataframe=pd.DataFrame(
            {
                "hash": pd.Series(rows["hash"], dtype=pd.ArrowDtype(pa.binary())),
                "left_id": pd.Series(rows["left_id"], dtype=pd.ArrowDtype(pa.binary())),
                "right_id": pd.Series(
                    rows["right_id"], dtype=pd.ArrowDtype(pa.binary())
                ),
                "probability": pd.Series(
                    rows["probability"], dtype=pd.ArrowDtype(pa.float32())
                ),
            }
        ),
        metadata=metadata,
    )


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
    source_info = _get_source_info(engine=engine, model_hash=model.hash)

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

    # Subquery to identify leaf nodes (clusters with no children)
    leaf_nodes = ~exists().where(Contains.parent == Clusters.hash)

    # Recursive CTE to get all descendants
    descendants = select(
        Contains.parent.label("component"),
        Contains.child.label("descendant"),
        literal(1).label("depth"),
    ).cte(recursive=True)

    descendants_recursive = descendants.union_all(
        select(
            descendants.c.component,
            Contains.child.label("descendant"),
            descendants.c.depth + 1,
        ).join(Contains, Contains.parent == descendants.c.descendant)
    )

    # Final query to get all components with their leaf descendants
    components_query = (
        select(
            Clusters.hash.label("parent"),
            descendants_recursive.c.descendant.label("child"),
            Probabilities.probability.label("threshold"),
        )
        .join(
            Probabilities,
            and_(
                Probabilities.cluster == Clusters.hash,
                Probabilities.model == hash_to_hex_decode(model.hash),
            ),
        )
        .join(descendants_recursive, descendants_recursive.c.component == Clusters.hash)
        .where(leaf_nodes)
        .order_by(Probabilities.probability.desc())
        .distinct()
    )

    return ClusterResults(
        dataframe=sql_to_df(stmt=components_query, engine=engine, return_type="pandas"),
        metadata=metadata,
    )
