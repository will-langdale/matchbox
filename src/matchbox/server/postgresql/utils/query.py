import logging
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pyarrow as pa
from pandas import ArrowDtype, DataFrame
from sqlalchemy import Engine, and_, func, select
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.exceptions import (
    MatchboxDatasetError,
    MatchboxModelError,
)
from matchbox.server.models import Probability, Source
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Models,
    ModelType,
    Probabilities,
    Sources,
)

if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
else:
    PolarsDataFrame = Any

T = TypeVar("T")

logic_logger = logging.getLogger("mb_logic")


def hash_to_hex_decode(hash: bytes) -> bytes:
    """A workround for PostgreSQL so we can compile the query and use ConnectorX."""
    return func.decode(hash.hex(), "hex")


def key_to_sqlalchemy_label(key: str, source: Source) -> str:
    """Converts a key to the SQLAlchemy LABEL_STYLE_TABLENAME_PLUS_COL."""
    return f"{source.db_schema}_{source.db_table}_{key}"


def _get_threshold_for_model(
    model: Models,
    model_hash: bytes,
    threshold: float | dict[str, float] | None = None,
    ancestor_model: Models | None = None,
) -> float:
    """Helper function to determine the appropriate threshold for a model.

    * If no threshold is specified, we use the model's truth value
    * If threshold is a float, use it for the main model, annd ancestors_cache for
        all ancestor models
    # If threshold is a dict, look up the threshold by model name

    Args:
        model: The main model object we're building thresholds for
        model_hash: Hash of the current model being processed
        threshold: The threshold parameter passed to the main function
        ancestor_model: If processing an ancestor, this is the ancestor model object

    Returns:
        float: The threshold value to use for this model
    """
    if threshold is None:
        return (
            model.truth
            if model_hash == model.hash
            else model.ancestors_cache.get(ancestor_model.hash.hex())
        )

    if isinstance(threshold, float):
        return (
            threshold
            if model_hash == model.hash
            else model.ancestors_cache.get(ancestor_model.hash.hex())
        )

    if isinstance(threshold, dict):
        target_model = ancestor_model or model
        return threshold.get(target_model.name, target_model.truth)

    raise ValueError(f"Invalid threshold type: {type(threshold)}")


def _get_lineage_clusters(
    model_lineage: list[bytes],
    dataset_hash: bytes,
    engine: Engine,
    threshold: float | dict[str, float] | None = None,
) -> Select:
    """Get the final cluster hash for each leaf node after traversing the lineage."""
    with Session(engine) as session:
        # Start with base mapping of IDs to their original clusters
        mapping = (
            select(
                Clusters.hash.label("cluster_hash"),
                func.unnest(Clusters.id).label("id"),
            )
            .where(
                and_(
                    Clusters.dataset == hash_to_hex_decode(dataset_hash),
                    Clusters.id.isnot(None),
                )
            )
            .cte("mapping_0")
        )

        # Process from oldest to newest
        for i, model_hash in enumerate(model_lineage, 1):
            model = (
                session.query(Models)
                .filter(Models.hash == hash_to_hex_decode(model_hash))
                .first()
            )

            if model.type == ModelType.DATASET.value:
                continue

            threshold_value = _get_threshold_for_model(
                model=model,
                model_hash=model_hash,
                threshold=threshold,
            )

            # Get valid merges for this model
            valid_merges = (
                select(
                    Contains.child.label("old_hash"),
                    Probabilities.cluster.label("new_hash"),
                )
                .join(
                    Contains,
                    and_(
                        Contains.parent == Probabilities.cluster,
                        Probabilities.model == hash_to_hex_decode(model_hash),
                        Probabilities.probability >= threshold_value,
                    ),
                )
                .cte(f"valid_merges_{i}")
            )

            # Update mapping with valid merges
            mapping = (
                select(
                    func.coalesce(
                        valid_merges.c.new_hash, mapping.c.cluster_hash
                    ).label("cluster_hash"),
                    mapping.c.id,
                )
                .select_from(mapping)
                .join(
                    valid_merges,
                    valid_merges.c.old_hash == mapping.c.cluster_hash,
                    isouter=True,
                )
                .cte(f"mapping_{i}")
            )

        # Final select of the mapping
        return select(mapping.c.cluster_hash.label("hash"), mapping.c.id)


def _model_to_hashes(
    dataset_hash: bytes,
    model_hash: bytes,
    engine: Engine,
    threshold: float | dict[str, float] | None = None,
) -> Select:
    """Takes a dataset model hash and model hash and returns all valid leaf clusters.

    * For dataset models, returns all clusters directly
    * For other models
        * Uses the probability filter to get valid clusters
        * Recursively traverses the Contains graph to get all leaf clusters

    Args:
        dataset_hash: The hash of the dataset model
        model_hash: The hash of the model to check
        engine: SQLAlchemy engine to use
        threshold: Threshold value to use for the model and its ancestors
    """
    with Session(engine) as session:
        model = (
            session.query(Models)
            .filter(Models.hash == hash_to_hex_decode(model_hash))
            .first()
        )
        if not model:
            raise MatchboxModelError()

        if model.type == ModelType.DATASET.value:
            return (
                select(
                    Clusters.hash.label("hash"),
                    func.unnest(Clusters.id).label("id"),
                )
                .distinct()
                .where(
                    and_(
                        Clusters.dataset == hash_to_hex_decode(dataset_hash),
                        Clusters.dataset.isnot(None),
                        Clusters.id.isnot(None),
                    )
                )
            )

    return _get_lineage_clusters(
        model_lineage=[ancestor.hash for ancestor in model.ancestors] + [model.hash],
        dataset_hash=dataset_hash,
        engine=engine,
        threshold=threshold,
    )


def query(
    selector: dict[Source, list[str]],
    engine: Engine,
    return_type: Literal["pandas", "arrow", "polars"] = "pandas",
    model: str | None = None,
    threshold: float | dict[str, float] | None = None,
    limit: int = None,
) -> DataFrame | pa.Table | PolarsDataFrame:
    """
    Queries Matchbox and the Source warehouse to retrieve linked data.

    Takes the dictionaries of tables and fields outputted by selectors and
    queries database for them. If a model "point of truth" is supplied, will
    attach the clusters this data belongs to.

    To accomplish this, the function:

    * Iterates through each selector, and
        * Retrieves its data in Matchbox according to the optional point of truth,
        including its hash and cluster hash
        * Retrieves its raw data from its Source's warehouse
        * Joins the two together
    * Unions the results, one row per item of data in the warehouses

    Returns:
        A table containing the requested data from each table, unioned together,
        with the hash key of each row in Matchbox, in the requested return type.
        If a model was given, also contains the hash cluster of each data --
        the entity the data belongs to according to the model.
    """
    tables: list[pa.Table] = []

    if limit:
        limit_base = limit // len(selector)
        limit_remainder = limit % len(selector)

    for source, fields in selector.items():
        with Session(engine) as session:
            # Get dataset
            get_dataset_model_stmt = (
                select(Models)
                .join(Sources, Sources.model == Models.hash)
                .where(
                    Sources.schema == source.db_schema,
                    Sources.table == source.db_table,
                    Sources.id == source.db_pk,
                )
            )
            dataset = session.execute(get_dataset_model_stmt).scalar_one_or_none()
            if dataset is None:
                raise MatchboxDatasetError(
                    db_schema=source.db_schema, db_table=source.db_table
                )

            if model is not None:
                model_hash = (
                    session.query(Models.hash).filter(Models.name == model).scalar()
                )
                if model_hash is None:
                    raise MatchboxModelError(model_name=model)
            else:
                model_hash = dataset.hash

            hash_query = _model_to_hashes(
                dataset_hash=dataset.hash,
                model_hash=model_hash,
                threshold=threshold,
                engine=engine,
            )

        if limit:
            remain = 0
            if limit_remainder:
                remain = 1
                limit_remainder -= 1
            hash_query = hash_query.limit(limit_base + remain)

        mb_hashes = sql_to_df(hash_query, engine, return_type="arrow")

        raw_data = source.to_arrow(
            fields=set([source.db_pk] + fields), pks=mb_hashes["id"].to_pylist()
        )

        joined_table = raw_data.join(
            right_table=mb_hashes,
            keys=key_to_sqlalchemy_label(key=source.db_pk, source=source),
            right_keys="id",
            join_type="inner",
        )

        keep_cols = ["hash"] + [key_to_sqlalchemy_label(f, source) for f in fields]
        match_cols = [col for col in joined_table.column_names if col in keep_cols]

        tables.append(joined_table.select(match_cols))

    result = pa.concat_tables(tables, promote_options="default")

    if return_type == "arrow":
        return result
    elif return_type == "pandas":
        return result.to_pandas(
            use_threads=True,
            split_blocks=True,
            self_destruct=True,
            types_mapper=ArrowDtype,
        )
    else:
        ValueError(f"return_type of {return_type} not valid")


def get_model_probabilities(engine: Engine, model_hash: bytes) -> set[Probability]:
    """
    Get all original probabilities proposed by a model.

    These are identified by:

    * Exactly two children
    * Both children are leaf nodes (not parents in Contains table)

    Orders children consistently based on their origin (dataset or proposing model).

    Args:
        model_hash: Hash of the model to query

    Returns:
        Set of Probability objects
    """
    with Session(engine) as session:
        Child = aliased(Clusters)
        ChildProb = aliased(Probabilities)

        # Subquery to find clusters that are parents in Contains
        parent_clusters = (select(Contains.parent).distinct()).scalar_subquery()

        # Main query
        query = (
            select(
                Clusters.hash,
                Child.hash.label("child_hash"),
                Child.dataset.label("source_dataset"),
                ChildProb.model.label("source_model"),
                Probabilities.probability,
            )
            .join(
                Probabilities,
                and_(
                    Probabilities.cluster == Clusters.hash,
                    Probabilities.model == model_hash,
                ),
            )
            # Join to get children
            .join(Contains, Contains.parent == Clusters.hash)
            .join(Child, Child.hash == Contains.child)
            # Left join to get potential source model
            .outerjoin(
                ChildProb,
                and_(ChildProb.cluster == Child.hash),
            )
            # Ensure children are leaf nodes
            .where(~Child.hash.in_(parent_clusters))
            # Only get clusters with exactly two children
            .having(func.count(Contains.child) == 2)
            .group_by(
                Clusters.hash,
                Child.hash,
                Child.dataset,
                ChildProb.model,
                Probabilities.probability,
            )
        )

        rows = session.execute(query).all()
        probabilities: set[Probability] = set()

        # Group by parent hash to pair children
        grouped = {}
        for row in rows:
            if row.hash not in grouped:
                grouped[row.hash] = []
            source_hash = (
                row.source_dataset
                if row.source_dataset is not None
                else row.source_model
            )
            grouped[row.hash].append((row.child_hash, source_hash))

        for parent_hash, children in grouped.items():
            if len(children) == 2:
                raise ValueError("Expected exactly two children")

            (left, _), (right, _) = sorted(children, key=lambda x: x[1])

            probabilities.add(
                Probability(
                    hash=parent_hash,
                    left=left,
                    right=right,
                    probability=rows[0].probability,  # Same for both rows
                )
            )

        return probabilities
