import logging

from sqlalchemy import (
    Engine,
    delete,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from matchbox.server.base import Cluster, Probability
from matchbox.server.exceptions import MatchboxDBDataError
from matchbox.server.postgresql.clusters import Clusters, clusters_association
from matchbox.server.postgresql.dedupe import DDupeContains, DDupeProbabilities, Dedupes
from matchbox.server.postgresql.link import LinkContains, LinkProbabilities, Links
from matchbox.server.postgresql.models import Models, ModelsFrom
from matchbox.server.postgresql.utils.db import batch_ingest
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


def insert_probabilities(
    model: str,
    engine: Engine,
    probabilities: list[Probability],
    batch_size: int,
    is_deduper: bool,
) -> None:
    """Writes probabilities to Matchbox."""
    probability_type = "Deduplication" if is_deduper else "Linking"
    metadata = f"{model} [{probability_type}]"

    if not probabilities:
        logic_logger.info(f"{metadata} No {probability_type.lower()} data to insert")
        return
    else:
        logic_logger.info(
            f"{metadata} Writing {probability_type.lower()} data "
            f"with batch size {batch_size}"
        )

    if is_deduper:
        ProbabilitiesTable = DDupeProbabilities
        NodesTable = Dedupes
    else:
        ProbabilitiesTable = LinkProbabilities
        NodesTable = Links

    with Session(engine) as session:
        # Get model
        db_model = session.query(Models).filter_by(name=model).first()
        model_sha1 = db_model.sha1

        if db_model is None:
            raise MatchboxDBDataError(source=Models, data=model)

        # Clear old model probabilities
        old_probs_subquery = (
            (db_model.proposes_dedupes if is_deduper else db_model.proposes_links)
            .select()
            .with_only_columns(ProbabilitiesTable.model)
        )

        session.execute(
            delete(ProbabilitiesTable).where(
                ProbabilitiesTable.model.in_(old_probs_subquery)
            )
        )

        session.commit()

        logic_logger.info(
            f"{metadata} Removed old {probability_type.lower()} probabilities"
        )

    with engine.connect() as conn:
        logic_logger.info(
            f"{metadata} Inserting %s {probability_type.lower()} objects",
            len(probabilities),
        )

        # Upsert nodes
        def probability_to_node(probability: Probability) -> dict:
            return {
                "sha1": probability.sha1,
                "left": probability.left,
                "right": probability.right,
            }

        batch_ingest(
            records=[probability_to_node(prob) for prob in probabilities],
            table=NodesTable,
            conn=conn,
            batch_size=batch_size,
        )

        # Insert probabilities
        def probability_to_probability(
            probability: Probability, model_sha1: bytes
        ) -> dict:
            return {
                "ddupe" if is_deduper else "link": probability.sha1,
                "model": model_sha1,
                "probability": probability.probability,
            }

        batch_ingest(
            records=[
                probability_to_probability(prob, model_sha1) for prob in probabilities
            ],
            table=ProbabilitiesTable,
            conn=conn,
            batch_size=batch_size,
        )

        logic_logger.info(
            f"{metadata} Inserted all %s {probability_type.lower()} objects",
            len(probabilities),
        )

    logic_logger.info(f"{metadata} Complete!")


def insert_clusters(
    model: str,
    engine: Engine,
    clusters: list[Cluster],
    batch_size: int,
    is_deduper: bool,
) -> None:
    """Writes clusters to Matchbox."""
    metadata = f"{model} [{'Deduplication' if is_deduper else 'Linking'}]"

    if not clusters:
        logic_logger.info(f"{metadata} No cluster data to insert")
        return
    else:
        logic_logger.info(
            f"{metadata} Writing cluster data with batch size {batch_size}"
        )

    Contains = DDupeContains if is_deduper else LinkContains

    with Session(engine) as session:
        # Get model
        db_model = session.query(Models).filter_by(name=model).first()
        model_sha1 = db_model.sha1

        if db_model is None:
            raise MatchboxDBDataError(source=Models, data=model)

        # Clear old model endorsements
        old_cluster_creates_subquery = db_model.creates.select().with_only_columns(
            Clusters.sha1
        )

        session.execute(
            delete(clusters_association).where(
                clusters_association.c.child.in_(old_cluster_creates_subquery)
            )
        )

        session.commit()

        logic_logger.info(f"{metadata} Removed old clusters")

    with engine.connect() as conn:
        logic_logger.info(
            f"{metadata} Inserting %s cluster objects",
            len(clusters),
        )

        # Upsert cluster nodes
        def cluster_to_cluster(cluster: Cluster) -> dict:
            """Prepares a Cluster for the Clusters table."""
            return {
                "sha1": cluster.parent,
            }

        batch_ingest(
            records=list({cluster_to_cluster(cluster) for cluster in clusters}),
            table=Clusters,
            conn=conn,
            batch_size=batch_size,
        )

        # Insert cluster contains
        def cluster_to_cluster_contains(cluster: Cluster) -> dict:
            """Prepares a Cluster for the Contains tables."""
            return {
                "parent": cluster.parent,
                "child": cluster.child,
            }

        batch_ingest(
            records=[cluster_to_cluster_contains(cluster) for cluster in clusters],
            table=Contains,
            conn=conn,
            batch_size=batch_size,
        )

        # Insert cluster proposed by
        def cluster_to_cluster_association(cluster: Cluster, model_sha1: bytes) -> dict:
            """Prepares a Cluster for the cluster association table."""
            return {
                "parent": model_sha1,
                "child": cluster.parent,
            }

        batch_ingest(
            records=[
                cluster_to_cluster_association(cluster, model_sha1)
                for cluster in clusters
            ],
            table=clusters_association,
            conn=conn,
            batch_size=batch_size,
        )

        logic_logger.info(
            f"{metadata} Inserted all %s cluster objects",
            len(clusters),
        )

    logic_logger.info(f"{metadata} Complete!")
