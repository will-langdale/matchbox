import logging
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pyarrow as pa
from pandas import ArrowDtype, DataFrame
from sqlalchemy import Engine, and_, func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.exceptions import MatchboxDatasetError, MatchboxModelError
from matchbox.server.models import Source
from matchbox.server.postgresql.orm import (
    Clusters,
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


def _get_model_lineage(model_hash: bytes, engine: Engine) -> Select:
    """Get all models in the lineage (ancestors + self) for a given model."""
    with Session(engine) as session:
        model = (
            session.query(Models)
            .filter(Models.hash == hash_to_hex_decode(model_hash))
            .first()
        )
        if not model:
            raise MatchboxModelError()

        ancestors = model.all_ancestors
        return [ancestor.hash for ancestor in ancestors] + [model.hash]


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
    """Get all valid clusters from the model lineage.

    A cluster is valid if it either:
    1. Belongs to the dataset, or
    2. Has a probability >= threshold from any model in the lineage
    """
    with Session(engine) as session:
        # Start with dataset's clusters
        dataset_clusters = select(Clusters.hash.label("cluster_hash")).where(
            Clusters.dataset == hash_to_hex_decode(dataset_hash)
        )

        # For each non-dataset model in lineage, add its valid clusters
        model_valid_clusters = []
        for model_hash in model_lineage:
            model = (
                session.query(Models)
                .filter(Models.hash == hash_to_hex_decode(model_hash))
                .first()
            )

            # Skip dataset models
            if model.type == ModelType.DATASET.value:
                continue

            # Check if model has probabilities
            has_probabilities = session.query(
                session.query(Probabilities)
                .filter(Probabilities.model == hash_to_hex_decode(model_hash))
                .exists()
            ).scalar()

            if has_probabilities:
                threshold_value = _get_threshold_for_model(
                    model=model,
                    model_hash=model_hash,
                    threshold=threshold,
                )
                if threshold_value is not None:
                    # Get clusters that meet threshold for this model
                    model_clusters = select(
                        Probabilities.cluster.label("cluster_hash")
                    ).where(
                        and_(
                            Probabilities.model == hash_to_hex_decode(model_hash),
                            Probabilities.probability >= threshold_value,
                        )
                    )
                    model_valid_clusters.append(model_clusters)

        # Combine dataset clusters with model clusters
        if model_valid_clusters:
            final_query = dataset_clusters.union(*model_valid_clusters)
            final_query = select(final_query.subquery().c.cluster_hash).distinct()
        else:
            final_query = dataset_clusters.distinct()

        return final_query.scalar_subquery()


def _model_to_hashes(
    dataset_hash: bytes,
    model_hash: bytes,
    engine: Engine,
    threshold: float | dict[str, float] | None = None,
) -> Select:
    """Takes a dataset model hash and model hash and returns all valid leaf clusters.

    1. Get the complete model lineage (ancestors + current model)
    2. Get all clusters that belong to this lineage and meet thresholds
    3. Find the leaf clusters from this set: data in the warehouse

    Handles dataset models outside of this logic.
    """
    # Handle dataset models
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
                select(Clusters.hash, func.unnest(Clusters.id).label("id"))
                .distinct()
                .where(
                    and_(
                        Clusters.dataset == hash_to_hex_decode(dataset_hash),
                        Clusters.dataset.isnot(None),
                        Clusters.id.isnot(None),
                    )
                )
            )

    model_lineage = _get_model_lineage(model_hash, engine)

    valid_clusters = _get_lineage_clusters(
        model_lineage=model_lineage,
        dataset_hash=dataset_hash,
        engine=engine,
        threshold=threshold,
    )

    return (
        select(Clusters.hash, func.unnest(Clusters.id).label("id"))
        .distinct()
        .where(
            and_(
                Clusters.hash.in_(valid_clusters),
                Clusters.dataset == hash_to_hex_decode(dataset_hash),
                Clusters.dataset.isnot(None),
                Clusters.id.isnot(None),
            )
        )
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

            # Get model
            if model is not None:
                model = session.query(Models).filter(Models.name == model).first()
                if model is None:
                    raise MatchboxModelError(model_name=model)
            else:
                # If no model is given, use the dataset's model
                model = dataset

            hash_query = _model_to_hashes(
                dataset_hash=dataset.hash,
                model_hash=model.hash,
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

        # Keep only the columns we want
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
