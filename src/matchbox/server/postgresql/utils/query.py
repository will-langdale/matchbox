import logging
from typing import Literal, TypeVar

import pyarrow as pa
from pandas import ArrowDtype, DataFrame
from sqlalchemy import Engine, Float, and_, cast, func, select, union
from sqlalchemy import text as sqltext
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
    Probabilities,
    Sources,
)

T = TypeVar("T")

logic_logger = logging.getLogger("mb_logic")


def hash_to_hex_decode(hash: bytes) -> bytes:
    """A workround for PostgreSQL so we can compile the query and use ConnectorX."""
    return func.decode(hash.hex(), "hex")


def key_to_sqlalchemy_label(key: str, source: Source) -> str:
    """Converts a key to the SQLAlchemy LABEL_STYLE_TABLENAME_PLUS_COL."""
    return f"{source.db_schema}_{source.db_table}_{key}"


def _build_probability_filter(model_hash: bytes, engine: Engine) -> Select:
    """Builds a filter subquery for probability thresholds based on model ancestors.

    Args:
        model_hash: Hash of the model to build filter for
    """
    # Get the model and its ancestors info
    model_info = (
        select(Models.ancestors, Models.truth, Models.hash).where(
            Models.hash == model_hash
        )
    ).subquery()

    # Compare ancestor truth values with those in JSON
    with Session(engine) as session:
        model = session.query(Models).filter(Models.hash == model_hash).first()

        if not model:
            raise MatchboxModelError()

        ancestors = model.all_ancestors()

        for ancestor in ancestors:
            ancestor_hash_hex = ancestor.hash.hex()
            json_threshold = model.ancestors.get(ancestor_hash_hex)
            if json_threshold is not None and ancestor.truth is not None:
                if abs(json_threshold - ancestor.truth) > 1e-6:  # Float comparison
                    logic_logger.warning(
                        f"Ancestor {ancestor.name} has truth value "
                        f"{ancestor.truth:f} but is specified as {json_threshold:f} in "
                        f"model {model.name}'s ancestors field. Using specified value "
                        f"{json_threshold:f}. \n\n"
                        "Use model.refresh_thresholds() to update the ancestors field."
                    )

    # Handle ancestors - compare against their specified threshold in ancestors JSONB
    ancestors_condition = (
        select(Probabilities.cluster)
        .join(Models, Probabilities.model == Models.hash)
        .where(
            and_(
                literal(True)
                == Models.hash.cast(sqltext("text")).in_(
                    select(func.jsonb_object_keys(model_info.c.ancestors))
                ),
                Probabilities.probability
                >= cast(
                    func.jsonb_extract_path_text(
                        model_info.c.ancestors, Models.hash.cast(sqltext("text"))
                    ),
                    Float,
                ),
            )
        )
    )

    # Handle the model itself - compare against its truth value
    model_condition = select(Probabilities.cluster).where(
        and_(
            Probabilities.model == model_hash,
            Probabilities.probability >= model_info.c.truth,
        )
    )

    # Combine into a single subquery of valid clusters
    valid_clusters = union(ancestors_condition, model_condition).scalar_subquery()

    return valid_clusters


def _model_to_hashes(dataset_hash: bytes, model_hash: bytes, engine: Engine) -> Select:
    """Takes a dataset model hash and model hash and returns all valid leaf clusters.

    Args:
        dataset_hash: The hash of the dataset model
        model_hash: The hash of the model to check
    """
    # Get valid clusters based on probability thresholds
    valid_clusters = _build_probability_filter(model_hash=model_hash, engine=engine)

    # Create recursive CTE to traverse the Contains graph
    recursive_clusters = (
        # Base case: start with valid clusters
        select([Contains.parent.label("start"), Contains.child.label("current")])
        .where(Contains.parent.in_(valid_clusters))
        .union(
            # Recursive case: follow children
            select([sqltext("start"), Contains.child]).select_from(
                sqltext("recursive_clusters").join(
                    Contains, sqltext("current") == Contains.parent
                )
            )
        )
    ).cte(recursive=True, name="recursive_clusters")

    # Final query to get leaf clusters
    query = (
        select(Clusters.hash, func.unnest(Clusters.id).label("id"))
        .distinct()
        .join(recursive_clusters, recursive_clusters.c.current == Clusters.hash)
        .where(Clusters.dataset == dataset_hash)
        .union(
            # Also include direct predictions on leaf clusters that meet threshold
            select(Clusters.hash, Clusters.id)
            .join(Probabilities, Clusters.hash == Probabilities.cluster)
            .where(
                and_(
                    Clusters.hash.in_(valid_clusters),
                    Clusters.dataset == dataset_hash,
                )
            )
        )
    )

    return query


def query(
    selector: dict[Source, list[str]],
    engine: Engine,
    return_type: Literal["pandas", "arrow"] = "pandas",
    model: str | None = None,
    limit: int = None,
) -> DataFrame | pa.Table:
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

    Args:
        selector: a dictionary with keys of table names, and values of a list
            of data required from that table, likely output by selector() or
            selectors()
        model (str): Optional. A model considered the point of truth to decide which
            clusters data belongs to
        return_type (str): the form to return data in, one of "pandas" or "arrow"
        engine: the SQLAlchemy engine to use
        limit (int): the number to use in a limit clause. Useful for testing

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
                dataset_hash=dataset.hash, model_hash=model.hash, engine=engine
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
