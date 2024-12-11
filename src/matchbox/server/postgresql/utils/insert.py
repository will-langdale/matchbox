import logging
from collections import defaultdict

from sqlalchemy import (
    Engine,
    delete,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from matchbox.common.db import Source
from matchbox.common.graph import ResolutionNodeKind
from matchbox.common.hash import dataset_to_hashlist, list_to_value_ordered_hash
from matchbox.common.results import ClusterResults, ProbabilityResults, Results
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
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

    resolution_hash = dataset.to_hash()

    resolution_data = {
        "hash": resolution_hash,
        "type": ResolutionNodeKind.DATASET.value,
        "name": f"{dataset.db_schema}.{dataset.db_table}",
    }

    source_data = {
        "resolution": resolution_hash,
        "schema": dataset.db_schema,
        "table": dataset.db_table,
        "id": dataset.db_pk,
    }

    clusters = dataset_to_hashlist(dataset=dataset, resolution_hash=resolution_hash)

    with engine.connect() as conn:
        logic_logger.info(f"Adding {dataset}")

        # Upsert into Resolutions table
        resolution_stmt = insert(Resolutions).values([resolution_data])
        resolution_stmt = resolution_stmt.on_conflict_do_update(
            index_elements=["hash"],
            set_={
                "name": resolution_stmt.excluded.name,
                "type": resolution_stmt.excluded.type,
            },
        )
        conn.execute(resolution_stmt)

        logic_logger.info(f"{dataset} added to Resolutions table")

        # Upsert into Sources table
        sources_stmt = insert(Sources).values([source_data])
        sources_stmt = sources_stmt.on_conflict_do_update(
            index_elements=["resolution"],
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
    left: Resolutions,
    right: Resolutions,
    description: str,
    engine: Engine,
) -> None:
    """
    Writes a model to Matchbox with a default truth value of 1.0.

    Args:
        model: Name of the new model
        left: Left parent of the model
        right: Right parent of the model. Same as left in a dedupe job
        description: Model description
        engine: SQLAlchemy engine instance

    Raises:
        MatchboxResolutionError if the specified parent models don't exist.

    Raises:
        MatchboxResolutionError if the specified model doesn't exist.
    """
    logic_logger.info(f"[{model}] Registering model")
    with Session(engine) as session:
        resolution_hash = list_to_value_ordered_hash(
            [left.hash, right.hash, bytes(model, encoding="utf-8")]
        )

        # Check if resolution exists
        exists_stmt = select(Resolutions).where(Resolutions.hash == resolution_hash)
        exists = session.scalar(exists_stmt) is not None

        # Upsert new resolution
        stmt = (
            insert(Resolutions)
            .values(
                hash=resolution_hash,
                type=ResolutionNodeKind.MODEL.value,
                name=model,
                description=description,
                truth=1.0,
            )
            .on_conflict_do_update(
                index_elements=["hash"],
                set_={"name": model, "description": description},
            )
        )

        session.execute(stmt)

        if not exists:

            def _create_closure_entries(parent_resolution: Resolutions) -> None:
                """Create closure entries for the new model, i.e. mappings between
                nodes and any of their direct or indirect parents"""
                session.add(
                    ResolutionFrom(
                        parent=parent_resolution.hash,
                        child=resolution_hash,
                        level=1,
                        truth_cache=parent_resolution.truth,
                    )
                )

                ancestor_entries = (
                    session.query(ResolutionFrom)
                    .filter(ResolutionFrom.child == parent_resolution.hash)
                    .all()
                )

                for entry in ancestor_entries:
                    session.add(
                        ResolutionFrom(
                            parent=entry.parent,
                            child=resolution_hash,
                            level=entry.level + 1,
                            truth_cache=entry.truth_cache,
                        )
                    )

            # Create resolution lineage entries
            _create_closure_entries(parent_resolution=left)

            if right != left:
                _create_closure_entries(parent_resolution=right)

        session.commit()

    status = "Inserted new" if not exists else "Updated existing"
    logic_logger.info(f"[{model}] {status} model with hash {resolution_hash}")
    logic_logger.info(f"[{model}] Done!")


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
    # Create initial hierarchy from base components
    prob_df = probabilities.dataframe
    cluster_df = clusters.dataframe

    thresholds = sorted(cluster_df["threshold"].unique(), reverse=True)

    # Add all clusters corresponding to a simple two-item probability edge
    hierarchy = []
    for _, row in prob_df.iterrows():
        parent, left_id, right_id, prob = row[
            ["hash", "left_id", "right_id", "probability"]
        ]
        hierarchy.extend(
            [(parent, left_id, float(prob)), (parent, right_id, float(prob))]
        )

    # Create adjacency structure for quick lookups
    adj_dict: dict[bytes, set[tuple[bytes, float]]] = defaultdict(set)
    for parent, child, prob in hierarchy:
        adj_dict[child].add((parent, prob))

    # Process each threshold level, getting clusters at each threshold
    for threshold in thresholds:
        threshold_float = float(threshold)

        current_clusters = cluster_df[cluster_df["threshold"] == threshold]

        # Group by parent to process each component
        for parent, group in current_clusters.groupby("parent"):
            members = set(group["child"])
            if len(members) <= 2:
                continue

            seen = set(members)
            current = set(members)
            ultimate_parents = set()

            # Keep traversing until we've explored all paths
            while current:
                next_level = set()
                # If any current nodes have no parents above threshold,
                # they are ultimate parents for this threshold
                for node in current:
                    parents = {
                        p for p, prob in adj_dict[node] if prob >= threshold_float
                    }
                    next_parents = parents - seen
                    if not parents:  # No parents = ultimate parent
                        ultimate_parents.add(node)

                    next_level.update(next_parents)
                    seen.update(parents)

                current = next_level

            for up in ultimate_parents:
                hierarchy.append((parent, up, threshold_float))
                adj_dict[up].add((parent, threshold_float))

    return sorted(hierarchy, key=lambda x: (x[2], x[0], x[1]), reverse=True)


def insert_results(
    resolution: Resolutions,
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
        resolution: Resolution of model kind to associate results with
        engine: SQLAlchemy engine instance
        results: A results object
        batch_size: Number of records to insert in each batch

    Raises:
        MatchboxResolutionError if the specified model doesn't exist.
    """
    logic_logger.info(
        f"[{resolution.name}] Writing results data with batch size {batch_size}"
    )

    with Session(engine) as session:
        try:
            # Clear existing probabilities for this resolution
            session.execute(
                delete(Probabilities).where(Probabilities.resolution == resolution.hash)
            )

            session.commit()
            logic_logger.info(f"[{resolution.name}] Removed old probabilities")

        except SQLAlchemyError as e:
            session.rollback()
            logic_logger.error(
                f"[{resolution.name}] Failed to clear old probabilities: {str(e)}"
            )
            raise

    with engine.connect() as conn:
        try:
            total_records = results.clusters.dataframe.shape[0]
            logic_logger.info(
                f"[{resolution.name}] Inserting {total_records} results objects"
            )

            cluster_records: list[tuple[bytes, None, None]] = []
            contains_records: list[tuple[bytes, bytes]] = []
            probability_records: list[tuple[bytes, bytes, float]] = []

            for parent, child, threshold in _cluster_results_to_hierarchical(
                probabilities=results.probabilities, clusters=results.clusters
            ):
                cluster_records.append((parent, None, None))
                contains_records.append((parent, child))
                probability_records.append((resolution.hash, parent, threshold))

            batch_ingest(
                records=cluster_records,
                table=Clusters,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{resolution.name}] Successfully inserted {len(cluster_records)} "
                "objects into Clusters table"
            )

            batch_ingest(
                records=contains_records,
                table=Contains,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{resolution.name}] Successfully inserted {len(contains_records)} "
                "objects into Contains table"
            )

            batch_ingest(
                records=probability_records,
                table=Probabilities,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{resolution.name}] Successfully inserted {len(probability_records)} "
                "objects into Probabilities table"
            )

        except SQLAlchemyError as e:
            logic_logger.error(f"[{resolution.name}] Failed to insert data: {str(e)}")
            raise

    logic_logger.info(f"[{resolution.name}] Insert operation complete!")
