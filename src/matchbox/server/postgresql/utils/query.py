import json
import logging
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pyarrow as pa
from pandas import ArrowDtype, DataFrame
from sqlalchemy import Engine, and_, func, select, union, union_all
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import literal
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.exceptions import MatchboxDatasetError, MatchboxModelError
from matchbox.server.models import Source
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


def _build_probability_filter(
    model_hash: bytes,
    engine: Engine,
    threshold: float | dict[str, float] | None = None,
) -> Select:
    """Builds a filter subquery for probability thresholds based on model ancestors.

    * Compares the ancestor truth values with those in the ancestor cache
    * Builds conditions for each ancestor model based on the threshold value

    Args:
        model_hash: Hash of the model to build filter for
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

        # For dataset models, return a query that selects all clusters
        if model.type == ModelType.DATASET.value:
            return select(Clusters.hash)

        # Compare w/ cache
        ancestors_cache = json.loads(model.ancestors_cache) or {}
        for ancestor in model.all_ancestors:
            ancestor_hash_hex = ancestor.hash.hex()
            json_threshold = ancestors_cache.get(ancestor_hash_hex)
            if json_threshold is not None and ancestor.truth is not None:
                if abs(json_threshold - ancestor.truth) > 1e-6:  # Float comparison
                    logic_logger.warning(
                        f"Ancestor {ancestor.name} has truth value "
                        f"{ancestor.truth:f} but is specified as {json_threshold:f} in "
                        f"model {model.name}'s ancestors_cache field. "
                        f"Using specified value {json_threshold:f}. \n\n"
                        "Use model.refresh_thresholds() to update the "
                        "ancestors_cache field."
                    )

        def build_ancestor_condition(ancestor: Models) -> Select:
            """Handles the ancestor model and a dynamic threshold value."""
            if ancestor.type == ModelType.DATASET.value:
                return select(Clusters.hash.label("cluster")).where(
                    Clusters.dataset == hash_to_hex_decode(ancestor.hash)
                )

            threshold_value = _get_threshold_for_model(
                model=model,
                model_hash=model_hash,
                threshold=threshold,
                ancestor_model=ancestor,
            )
            if threshold_value is None:
                raise ValueError(
                    f"No cached or specified threshold value found for {ancestor.name}"
                )

            return select(Probabilities.cluster.label("cluster")).where(
                and_(
                    Probabilities.model == hash_to_hex_decode(ancestor.hash),
                    Probabilities.probability >= threshold_value,
                )
            )

        # Combine all ancestor conditions
        if not model.all_ancestors:
            ancestors_condition = select(
                literal(None, type_=Probabilities.cluster.type).label("cluster")
            ).where(literal(False))
        else:
            ancestors_condition = union(
                *[
                    build_ancestor_condition(ancestor)
                    for ancestor in model.all_ancestors
                ]
            )

        # Handle the model itself with its threshold
        model_threshold = _get_threshold_for_model(model, model_hash, threshold)
        if model_threshold is None:
            model_condition = select(
                literal(None, type_=Probabilities.cluster.type).label("cluster")
            ).where(literal(False))
        else:
            model_condition = select(Probabilities.cluster.label("cluster")).where(
                and_(
                    Probabilities.model == hash_to_hex_decode(model_hash),
                    Probabilities.probability >= model_threshold,
                )
            )

        # Combine into a single subquery of valid clusters
        all_clusters = union_all(ancestors_condition, model_condition).subquery()

        return select(all_clusters.c.cluster).distinct().scalar_subquery()


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
                .where(Clusters.dataset == hash_to_hex_decode(dataset_hash))
            )

    # Get valid clusters over threshold
    valid_clusters = _build_probability_filter(
        model_hash=model_hash, engine=engine, threshold=threshold
    )

    recursive = (
        # Base case: start with valid clusters
        select(
            Contains.parent.label("start"),
            Contains.child.label("current"),
        )
        .where(Contains.parent.in_(valid_clusters))
        .cte(recursive=True, name="recursive")
    )

    # Add the recursive part
    recursive = recursive.union(
        select(
            recursive.c.start,
            Contains.child,
        ).join(Contains, Contains.parent == recursive.c.current)
    )

    # Final query to get leaf clusters
    query = (
        select(Clusters.hash, func.unnest(Clusters.id).label("id"))
        .distinct()
        .join(recursive, recursive.c.current == Clusters.hash)
        .where(Clusters.dataset == hash_to_hex_decode(dataset_hash))
        .union(
            # Also include direct predictions on leaf clusters that meet threshold
            select(Clusters.hash, func.unnest(Clusters.id).label("id"))
            .join(Probabilities, Clusters.hash == Probabilities.cluster)
            .where(
                and_(
                    Clusters.hash.in_(valid_clusters),
                    Clusters.dataset == hash_to_hex_decode(dataset_hash),
                )
            )
        )
    )

    return query


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
