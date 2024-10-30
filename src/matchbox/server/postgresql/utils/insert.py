import logging

from sqlalchemy import (
    Engine,
    delete,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from matchbox.common.hash import dataset_to_hashlist, list_to_value_ordered_hash
from matchbox.common.results import ClusterResults, Results
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
    """Writes a model to Matchbox with a default truth value of 1.0.

    Args:
        model: Name of the new model
        left: Name of the left parent model
        right: Name of the left parent model. Same as left in a link job
        description: Model description
        engine: SQLAlchemy engine instance

    Raises:
        MatchboxModelError if the specified parent models don't exist.

    Raises:
        MatchboxModelError if the specified model doesn't exist.
    """
    logic_logger.info(f"[{model}] Registering model")
    with Session(engine) as session:
        model_hash = list_to_value_ordered_hash([left.hash, right.hash])

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
            """Create closure entries for the new model."""
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


def _cluster_results_to_hierarchical(
    clusters: ClusterResults,
) -> list[tuple[bytes, bytes, float]]:
    """
    Converts a Results object to a more efficient hierarchical structure for PostgreSQL.

    * Two-item components are given a threshold of their original pairwise probability
    * Larger components are stored in a hierarchical structure, where if their children
        are a known component at a higher threshold, they reference that component

    This allows all results to be recovered from the database, albeit inefficiently,
    but allows simple and efficient querying of clusters at any threshold.

    This function requires that:

    * ClusterResults are sorted by threshold descending
    * Two-item components thresholds are the original pairwise probabilities

    Args:
        components_df: DataFrame with parent, child, threshold from to_components()
        original_df: Original DataFrame with left_id, right_id, probability

    Returns:
        A tuple of (parent, child, threshold) ready for insertion
    """
    parents = []
    children = []
    thresholds = []

    # hash -> (threshold, is_component)
    component_info: dict[bytes, tuple[float, bool]] = {}

    # Process components in descending threshold order
    for threshold, group in clusters.dataframe.groupby("threshold", sort=True):
        current_components = set()

        # Process all parents at this threshold at once
        for parent, parent_children in group.groupby("parent")["child"]:
            child_hashes = frozenset(parent_children)

            # Partition children into original and subcomponents
            original = []
            subcomponents = []

            for child in child_hashes:
                if child in component_info:
                    prev_threshold, is_comp = component_info[child]
                    if prev_threshold >= threshold and is_comp:
                        subcomponents.append(child)
                        continue
                original.append(child)

            parents.extend([parent] * len(original))
            children.extend(original)
            thresholds.extend([threshold] * len(original))

            parents.extend([parent] * len(subcomponents))
            children.extend(subcomponents)
            thresholds.extend([threshold] * len(subcomponents))

            # Mark this parent as a component
            component_info[parent] = (threshold, True)
            current_components.add(parent)

            # Mark original children as non-components at this threshold
            for child in original:
                if child not in component_info:
                    component_info[child] = (threshold, False)

    return list(zip(parents, children, thresholds, strict=True))


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
                f"[{model.name}] Inserting {total_records} probability objects"
            )

            cluster_records: list[tuple[bytes, None, None]] = []
            contains_records: list[tuple[bytes, bytes]] = []
            probability_records: list[tuple[bytes, bytes, float]] = []

            for parent, child, threshold in _cluster_results_to_hierarchical(
                clusters=results.clusters
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
