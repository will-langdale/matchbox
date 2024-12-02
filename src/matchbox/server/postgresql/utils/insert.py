import logging

import rustworkx as rx
from sqlalchemy import (
    Engine,
    delete,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from matchbox.common.hash import dataset_to_hashlist, list_to_value_ordered_hash
from matchbox.common.results import ClusterResults, ProbabilityResults, Results
from matchbox.server.models import Source
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Models,
    ModelsFrom,
    ModelType,
    Probabilities,
    Sources,
)
from matchbox.server.postgresql.utils.db import batch_ingest

logic_logger = logging.getLogger("mb_logic")


def insert_dataset(dataset: Source, engine: Engine, batch_size: int) -> None:
    """Indexes a dataset from your data warehouse within Matchbox."""

    db_logger = logging.getLogger("sqlalchemy.engine")
    db_logger.setLevel(logging.WARNING)

    ##################
    # Insert dataset #
    ##################

    model_hash = dataset.to_hash()

    model_data = {
        "hash": model_hash,
        "type": ModelType.DATASET.value,
        "name": f"{dataset.db_schema}.{dataset.db_table}",
    }

    source_data = {
        "model": model_hash,
        "schema": dataset.db_schema,
        "table": dataset.db_table,
        "id": dataset.db_pk,
    }

    clusters = dataset_to_hashlist(dataset=dataset, model_hash=model_hash)

    with engine.connect() as conn:
        logic_logger.info(f"Adding {dataset}")

        # Upsert into Models table
        models_stmt = insert(Models).values([model_data])
        models_stmt = models_stmt.on_conflict_do_update(
            index_elements=["hash"],
            set_={
                "name": models_stmt.excluded.name,
                "type": models_stmt.excluded.type,
            },
        )
        conn.execute(models_stmt)

        logic_logger.info(f"{dataset} added to Models table")

        # Upsert into Sources table
        sources_stmt = insert(Sources).values([source_data])
        sources_stmt = sources_stmt.on_conflict_do_update(
            index_elements=["model"],
            set_={
                "schema": sources_stmt.excluded.schema,
                "table": sources_stmt.excluded.table,
                "id": sources_stmt.excluded.id,
            },
        )
        conn.execute(sources_stmt)

        conn.commit()

        logic_logger.info(f"{dataset} added to Sources table")

        # Upsert into Clusters table
        batch_ingest(
            records=[(clus["hash"], clus["dataset"], clus["id"]) for clus in clusters],
            table=Clusters,
            conn=conn,
            batch_size=batch_size,
        )

        conn.commit()

        logic_logger.info(f"{dataset} added {len(clusters)} objects to Clusters table")

    logic_logger.info(f"Finished {dataset}")


def insert_model(
    model: str,
    left: Models,
    right: Models,
    description: str,
    engine: Engine,
) -> None:
    """
    Writes a model to Matchbox with a default truth value of 1.0.

    Args:
        model: Name of the new model
        left: Name of the left parent model
        right: Name of the right parent model. Same as left in a link job
        description: Model description
        engine: SQLAlchemy engine instance

    Raises:
        MatchboxModelError if the specified parent models don't exist.

    Raises:
        MatchboxModelError if the specified model doesn't exist.
    """
    logic_logger.info(f"[{model}] Registering model")
    with Session(engine) as session:
        model_hash = list_to_value_ordered_hash([left.hash, right.hash, model])

        # Create new model
        new_model = Models(
            hash=model_hash,
            type=ModelType.MODEL.value,
            name=model,
            description=description,
            truth=1.0,
        )
        session.add(new_model)
        session.flush()

        def _create_closure_entries(parent_model: Models) -> None:
            """Create closure entries for the new model, i.e. mappings between
            nodes and any of their direct or indirect parents"""
            session.add(
                ModelsFrom(
                    parent=parent_model.hash,
                    child=model_hash,
                    level=1,
                    truth_cache=parent_model.truth,
                )
            )

            ancestor_entries = (
                session.query(ModelsFrom)
                .filter(ModelsFrom.child == parent_model.hash)
                .all()
            )

            for entry in ancestor_entries:
                session.add(
                    ModelsFrom(
                        parent=entry.parent,
                        child=model_hash,
                        level=entry.level + 1,
                        truth_cache=entry.truth_cache,
                    )
                )

        # Create model lineage entries
        _create_closure_entries(parent_model=left)

        if right != left:
            _create_closure_entries(parent_model=right)

        session.commit()

    logic_logger.info(f"[{model}] Done!")


def _find_ultimate_parents(subgraph: rx.PyDiGraph, child_nodes: set[int]) -> set[int]:
    """Find ultimate parents of the child nodes in the subgraph."""
    all_ancestors = set().union(
        *(rx.ancestors(subgraph, child) for child in child_nodes)
    )

    return {node for node in all_ancestors if len(subgraph.in_edges(node)) == 0}


def _cluster_results_to_hierarchical(
    probabilities: ProbabilityResults,
    clusters: ClusterResults,
) -> list[tuple[bytes, bytes, float]]:
    """
    Converts results to a hierarchical structure by building up from base components.

    Args:
        probabilities: Original pairwise probabilities containing base components
        clusters: Connected components at each threshold

    Returns:
        List of (parent, child, threshold) tuples representing the hierarchy
    """
    # Create initial graph of base components
    graph = rx.PyDiGraph()
    nodes: dict[bytes, int] = {}  # node_name -> node_id
    hierarchy: list[tuple[bytes, bytes, float]] = []

    def get_node_id(name: bytes) -> int:
        if name not in nodes:
            nodes[name] = graph.add_node(name)
        return nodes[name]

    # 1. Build base component graph from ProbabilityResults
    prob_df = probabilities.dataframe
    for _, row in prob_df.iterrows():
        parent = row["hash"]
        left_id = row["left_id"]
        right_id = row["right_id"]
        prob = float(row["probability"])

        parent_id = get_node_id(parent)
        left_node = get_node_id(left_id)
        right_node = get_node_id(right_id)
        graph.add_edge(parent_id, left_node, prob)
        graph.add_edge(parent_id, right_node, prob)

        hierarchy.extend([(parent, left_id, prob), (parent, right_id, prob)])

    # 2. Process ClusterResults by threshold descending
    thresholds = sorted(clusters.dataframe["threshold"].unique(), reverse=True)

    for threshold in thresholds:
        group = clusters.dataframe[clusters.dataframe["threshold"] == threshold]
        threshold = float(threshold)

        # Create subgraph of relevant nodes and edges at this threshold
        graph_edge_indices = graph.edge_indices()
        subgraph_edges = [
            graph.get_edge_endpoints_by_index(graph_edge_indices[e])
            for e in graph.filter_edges(lambda w, t=threshold: w >= t)
        ]
        subgraph = graph.edge_subgraph(subgraph_edges)

        # Process each component at this threshold
        for parent, comp_group in group.groupby("parent"):
            members = set(comp_group["child"])
            if len(members) <= 2:
                continue

            # Find ultimate parents of children using threshold
            child_node_ids = {nodes[child] for child in members}
            ultimate_parent_ids = _find_ultimate_parents(
                subgraph=subgraph, child_nodes=child_node_ids
            )

            # Add component to graph
            parent_id = get_node_id(parent)

            # Add edges to ultimate parents
            for up_id in ultimate_parent_ids:
                up_name = graph.get_node_data(up_id)
                graph.add_edge(parent_id, up_id, threshold)
                hierarchy.append((parent, up_name, threshold))

    return sorted(hierarchy, key=lambda x: (x[2], x[0], x[1]), reverse=True)


def insert_results(
    model: Models,
    engine: Engine,
    results: Results,
    batch_size: int,
) -> None:
    """
    Writes a Results object to Matchbox.

    The PostgreSQL backend stores clusters in a hierarchical structure, where
    each component references its parent component at a higher threshold.

    This means two-item components are synonymous with their original pairwise
    probabilities.

    This allows easy querying of clusters at any threshold.

    Args:
        model: Model object to associate results with
        engine: SQLAlchemy engine instance
        results: A results object
        batch_size: Number of records to insert in each batch

    Raises:
        MatchboxModelError if the specified model doesn't exist.
    """
    logic_logger.info(
        f"[{model.name}] Writing results data with batch size {batch_size}"
    )

    with Session(engine) as session:
        try:
            # Clear existing probabilities for this model
            session.execute(
                delete(Probabilities).where(Probabilities.model == model.hash)
            )

            session.commit()
            logic_logger.info(f"[{model.name}] Removed old probabilities")

        except SQLAlchemyError as e:
            session.rollback()
            logic_logger.error(
                f"[{model.name}] Failed to clear old probabilities: {str(e)}"
            )
            raise

    with engine.connect() as conn:
        try:
            total_records = results.clusters.dataframe.shape[0]
            logic_logger.info(
                f"[{model.name}] Inserting {total_records} results objects"
            )

            cluster_records: list[tuple[bytes, None, None]] = []
            contains_records: list[tuple[bytes, bytes]] = []
            probability_records: list[tuple[bytes, bytes, float]] = []

            for parent, child, threshold in _cluster_results_to_hierarchical(
                probabilities=results.probabilities, clusters=results.clusters
            ):
                cluster_records.append((parent, None, None))
                contains_records.append((parent, child))
                probability_records.append((model.hash, parent, threshold))

            batch_ingest(
                records=cluster_records,
                table=Clusters,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{model.name}] Successfully inserted {len(cluster_records)} "
                "objects into Clusters table"
            )

            batch_ingest(
                records=contains_records,
                table=Contains,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{model.name}] Successfully inserted {len(contains_records)} "
                "objects into Contains table"
            )

            batch_ingest(
                records=probability_records,
                table=Probabilities,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{model.name}] Successfully inserted {len(probability_records)} "
                "objects into Probabilities table"
            )

        except SQLAlchemyError as e:
            logic_logger.error(f"[{model.name}] Failed to insert data: {str(e)}")
            raise

    logic_logger.info(f"[{model.name}] Insert operation complete!")
