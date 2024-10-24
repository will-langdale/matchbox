import json
import logging

from sqlalchemy import (
    Engine,
    delete,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from matchbox.common.exceptions import MatchboxModelError
from matchbox.common.hash import dataset_to_hashlist, list_to_value_ordered_hash
from matchbox.server.models import Probability, Source
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
        "ancestors_cache": {},
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
                "ancestors_cache": models_stmt.excluded.ancestors_cache,
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
    model: str, left: str, description: str, engine: Engine, right: str | None = None
) -> None:
    """Writes a model to Matchbox with a default truth value of 1.0.

    Raises:
        MatchboxModelError if the specified model doesn't exist.
    """
    logic_logger.info(f"[{model}] Registering model")
    with Session(engine) as session:
        left_model = session.query(Models).filter(Models.name == left).first()
        if not left_model:
            raise MatchboxModelError(model_name=left)

        # Overwritten with actual right model if in a link job
        right_model = left_model
        if right:
            right_model = session.query(Models).filter(Models.name == right).first()
            if not right_model:
                raise MatchboxModelError(model_name=right)

        model_hash = list_to_value_ordered_hash([left_model.hash, right_model.hash])

        # Calculate ancestors dictionary
        ancestors = {}
        if left_model.ancestors_cache:
            ancestors.update(json.loads(left_model.ancestors_cache))
        if right_model.ancestors_cache:
            ancestors.update(json.loads(right_model.ancestors_cache))

        ancestors[left_model.hash.hex()] = left_model.truth
        ancestors[right_model.hash.hex()] = right_model.truth

        # Create new model
        new_model = Models(
            hash=model_hash,
            type=ModelType.MODEL.value,
            name=model,
            description=description,
            truth=1.0,
            ancestors_cache=json.dumps(ancestors),
        )
        session.add(new_model)

        # Create model lineage entries
        parent_link = ModelsFrom(parent=left_model.hash, child=model_hash)
        session.add(parent_link)

        if right_model != left_model:
            right_link = ModelsFrom(parent=right_model.hash, child=model_hash)
            session.add(right_link)

        session.commit()

    logic_logger.info(f"[{model}] Done!")


def insert_probabilities(
    model: str,
    engine: Engine,
    probabilities: list[Probability],
    batch_size: int,
) -> None:
    """
    Writes probabilities and their associated clusters to Matchbox.

    Args:
        model: Name of the model to associate probabilities with
        engine: SQLAlchemy engine instance
        probabilities: List of Probability objects to insert
        batch_size: Number of records to insert in each batch

    Raises:
        MatchboxModelError if the specified model doesn't exist.
    """
    logic_logger.info(f"{model} Writing probability data with batch size {batch_size}")

    with Session(engine) as session:
        db_model = session.query(Models).filter_by(name=model).first()
        if db_model is None:
            raise MatchboxModelError(model_name=model)

        model_hash = db_model.hash

        try:
            # Clear existing probabilities for this model
            session.execute(
                delete(Probabilities).where(Probabilities.model == model_hash)
            )

            session.commit()
            logic_logger.info(f"{model} Removed old probabilities")

        except SQLAlchemyError as e:
            session.rollback()
            logic_logger.error(f"{model} Failed to clear old probabilities: {str(e)}")
            raise

    with engine.connect() as conn:
        try:
            total_records = len(probabilities)
            logic_logger.info(f"{model} Inserting {total_records} probability objects")

            cluster_records = [(prob.hash,) for prob in probabilities]

            batch_ingest(
                records=cluster_records,
                table=Clusters,
                conn=conn,
                batch_size=batch_size,
            )

            contains_records = []
            for prob in probabilities:
                contains_records.extend(
                    [
                        (prob.hash, prob.left),
                        (prob.hash, prob.right),
                    ]
                )

            batch_ingest(
                records=contains_records,
                table=Contains,
                conn=conn,
                batch_size=batch_size,
            )

            probability_records = [
                (model_hash, prob.hash, prob.probability) for prob in probabilities
            ]

            batch_ingest(
                records=probability_records,
                table=Probabilities,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"{model} Successfully inserted {total_records} "
                "probability objects and their associated clusters"
            )

        except SQLAlchemyError as e:
            logic_logger.error(f"{model} Failed to insert data: {str(e)}")
            raise

    logic_logger.info(f"{model} Insert operation complete!")
