import logging

from pg_bulk_ingest import Delete, Upsert, ingest
from sqlalchemy import (
    Engine,
    delete,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from matchbox.server.base import Probability
from matchbox.server.exceptions import MatchboxDBDataError
from matchbox.server.postgresql.dedupe import DDupeProbabilities, Dedupes
from matchbox.server.postgresql.link import LinkProbabilities, Links
from matchbox.server.postgresql.models import Models, ModelsFrom
from matchbox.server.postgresql.utils import data_to_batch
from matchbox.server.postgresql.utils.sha1 import (
    list_to_value_ordered_sha1,
    model_name_to_sha1,
)

logic_logger = logging.getLogger("mb_logic")


def insert_deduper(
    model: str, deduplicates: bytes, description: str, engine: Engine
) -> None:
    """Writes a deduper model to Matchbox."""
    metadata = f"{model} [Deduplication]"
    logic_logger.info(f"{metadata} Registering model")

    with Session(engine) as session:
        # Construct model SHA1 from name and what it deduplicates
        model_sha1 = list_to_value_ordered_sha1([model, deduplicates])

        # Insert model
        model = Models(
            sha1=model_sha1,
            name=model,
            description=description,
            deduplicates=deduplicates,
        )

        session.merge(model)
        session.commit()

        # Insert reference to parent models


def insert_linker(
    model: str, left: str, right: str, description: str, engine: Engine
) -> None:
    """Writes a linker model to Matchbox."""
    metadata = f"{model} [Linking]"
    logic_logger.info(f"{metadata} Registering model")

    with Session(engine) as session:
        # Construct model SHA1 from parent model SHA1s
        left_sha1 = model_name_to_sha1(left, engine=engine)
        right_sha1 = model_name_to_sha1(right, engine=engine)

        model_sha1 = list_to_value_ordered_sha1(
            [bytes(model, encoding="utf-8"), left_sha1, right_sha1]
        )

        # Insert model
        model = Models(
            sha1=model_sha1,
            name=model,
            description=description,
            deduplicates=None,
        )

        session.merge(model)
        session.commit()

        # Insert reference to parent models
        models_from_to_insert = [
            {"parent": model_sha1, "child": left_sha1},
            {"parent": model_sha1, "child": right_sha1},
        ]

        ins_stmt = insert(ModelsFrom)
        ins_stmt = ins_stmt.on_conflict_do_nothing(
            index_elements=[
                ModelsFrom.parent,
                ModelsFrom.child,
            ]
        )
        session.execute(ins_stmt, models_from_to_insert)
        session.commit()


def insert_deduplication_probabilities(
    model: str,
    engine: Engine,
    probabilities: list[Probability],
    batch_size: int,
) -> None:
    """Writes deduplication probabilities to Matchbox."""
    metadata = f"{model} [Deduplication]"

    if not len(probabilities):
        logic_logger.info(f"{metadata} No deduplication data to insert")
        return
    else:
        logic_logger.info(
            f"{metadata} Writing deduplication data with batch size {batch_size}"
        )

    def probability_to_ddupe(probability: Probability) -> dict:
        """Prepares a Probability for the Dedupes table."""
        return {
            "sha1": probability.sha1,
            "left": probability.left,
            "right": probability.right,
        }

    def probability_to_ddupeprobability(
        probability: Probability, model_sha1: bytes
    ) -> dict:
        """Prepares a Probability for the DDupeProbabilities table."""
        return {
            "ddupe": probability.sha1,
            "model": model_sha1,
            "probability": probability.probability,
        }

    with Session(engine) as session:
        # Add probabilities
        # Get model
        db_model = session.query(Models).filter_by(name=model).first()
        model_sha1 = model.sha1

        if db_model is None:
            raise MatchboxDBDataError(source=Models, data=model)

        # Clear old model probabilities
        old_ddupe_probs_subquery = db_model.proposes_dedupes.select().with_only_columns(
            DDupeProbabilities.model
        )

        session.execute(
            delete(DDupeProbabilities).where(
                DDupeProbabilities.model.in_(old_ddupe_probs_subquery)
            )
        )

        session.commit()

        logic_logger.info(f"{metadata} Removed old deduplication probabilities")

    with engine.connect() as conn:
        logic_logger.info(
            f"{metadata} Inserting %s deduplication objects",
            len(probabilities),
        )

        # Upsert dedupe nodes
        # Create data batching function and pass it to ingest
        fn_dedupe_batch = data_to_batch(
            records=[probability_to_ddupe(prob) for prob in probabilities],
            table=Dedupes.__table__,
            batch_size=batch_size,
        )

        ingest(
            conn=conn,
            metadata=Dedupes.metadata,
            batches=fn_dedupe_batch,
            upsert=Upsert.IF_PRIMARY_KEY,
            delete=Delete.OFF,
        )

        # Insert dedupe probabilities
        fn_dedupe_probs_batch = data_to_batch(
            records=[
                probability_to_ddupeprobability(prob, model_sha1)
                for prob in probabilities
            ],
            table=DDupeProbabilities.__table__,
            batch_size=batch_size,
        )

        ingest(
            conn=conn,
            metadata=DDupeProbabilities.metadata,
            batches=fn_dedupe_probs_batch,
            upsert=Upsert.IF_PRIMARY_KEY,
            delete=Delete.OFF,
        )

        logic_logger.info(
            f"{metadata} Inserted all %s deduplication objects",
            len(probabilities),
        )

    logic_logger.info(f"{metadata} Complete!")


def insert_link_probabilities(
    model: str,
    engine: Engine,
    probabilities: list[Probability],
    batch_size: int,
) -> None:
    """Writes link probabilities to Matchbox."""
    metadata = f"{model} [Linking]"

    if not len(probabilities):
        logic_logger.info(f"{metadata} No link data to insert")
        return
    else:
        logic_logger.info(f"{metadata} Writing link data with batch size {batch_size}")

    def probability_to_link(probability: Probability) -> dict:
        """Prepares a Probability for the Links table."""
        return {
            "sha1": probability.sha1,
            "left": probability.left,
            "right": probability.right,
        }

    def probability_to_linkprobability(
        probability: Probability, model_sha1: bytes
    ) -> dict:
        """Prepares a Probability for the LinkProbabilities table."""
        return {
            "link": probability.sha1,
            "model": model_sha1,
            "probability": probability.probability,
        }

    with Session(engine) as session:
        # Add probabilities
        # Get model
        model = session.query(Models).filter_by(name=model).first()
        model_sha1 = model.sha1

        if model is None:
            raise MatchboxDBDataError(source=Models, data=model)

        # Clear old model probabilities
        old_link_probs_subquery = model.proposes_links.select().with_only_columns(
            LinkProbabilities.model
        )

        session.execute(
            delete(LinkProbabilities).where(
                LinkProbabilities.model.in_(old_link_probs_subquery)
            )
        )

        session.commit()

        logic_logger.info(f"{metadata} Removed old link probabilities")

    with engine.connect() as conn:
        logic_logger.info(
            f"{metadata} Inserting %s link objects",
            len(probabilities),
        )

        # Upsert link nodes
        # Create data batching function and pass it to ingest
        fn_link_batch = data_to_batch(
            records=[probability_to_link(prob) for prob in probabilities],
            table=Links.__table__,
            batch_size=batch_size,
        )

        ingest(
            conn=conn,
            metadata=Links.metadata,
            batches=fn_link_batch,
            upsert=Upsert.IF_PRIMARY_KEY,
            delete=Delete.OFF,
        )

        # Insert link probabilities
        fn_link_probs_batch = data_to_batch(
            records=[
                probability_to_linkprobability(prob, model_sha1)
                for prob in probabilities
            ],
            table=LinkProbabilities.__table__,
            batch_size=batch_size,
        )

        ingest(
            conn=conn,
            metadata=LinkProbabilities.metadata,
            batches=fn_link_probs_batch,
            upsert=Upsert.IF_PRIMARY_KEY,
            delete=Delete.OFF,
        )

        logic_logger.info(
            f"{metadata} Inserted all %s link objects",
            len(probabilities),
        )

    logic_logger.info(f"{metadata} Complete!")
