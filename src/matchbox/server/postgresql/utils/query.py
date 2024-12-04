import logging
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pyarrow as pa
from pandas import ArrowDtype, DataFrame
from sqlalchemy import (
    Engine,
    and_,
    cast,
    func,
    literal,
    null,
    select,
    union,
)
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import get_schema_table_names, sql_to_df
from matchbox.common.exceptions import (
    MatchboxDatasetError,
    MatchboxModelError,
)
from matchbox.server.models import Match, Source
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Models,
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


def _resolve_thresholds(
    lineage_truths: dict[str, float],
    model: Models,
    threshold: float | dict[str, float] | None,
    session: Session,
) -> dict[bytes, float]:
    """
    Resolves final thresholds for each model in the lineage based on user input.

    Args:
        lineage_truths: Dict from get_lineage_to_dataset with model hash -> cached truth
        model: The target model being used for clustering
        threshold: User-supplied threshold value or dict
        session: SQLAlchemy session

    Returns:
        Dict mapping model hash to their final threshold values
    """
    resolved_thresholds = {}

    for model_hash, default_truth in lineage_truths.items():
        if threshold is None:
            resolved_thresholds[model_hash] = default_truth
        elif isinstance(threshold, float):
            resolved_thresholds[model_hash] = (
                threshold if model_hash == model.hash.hex() else default_truth
            )
        elif isinstance(threshold, dict):
            model_obj = (
                session.query(Models)
                .filter(Models.hash == bytes.fromhex(model_hash))
                .first()
            )
            resolved_thresholds[model_hash] = threshold.get(
                model_obj.name, default_truth
            )
        else:
            raise ValueError(f"Invalid threshold type: {type(threshold)}")

    return resolved_thresholds


def _get_valid_clusters_for_model(model_hash: bytes, threshold: float) -> Select:
    """Get clusters that meet the threshold for a specific model."""
    return select(Probabilities.cluster.label("cluster")).where(
        and_(
            Probabilities.model == hash_to_hex_decode(model_hash),
            Probabilities.probability >= threshold,
        )
    )


def _union_valid_clusters(lineage_thresholds: dict[bytes, float]) -> Select:
    """Creates a CTE of clusters that are valid for any model in the lineage.

    Each model may have a different threshold.
    """
    valid_clusters = None

    for model_hash, threshold in lineage_thresholds.items():
        if threshold is None:
            # This is a dataset - get all its clusters directly
            model_valid = select(Clusters.hash.label("cluster")).where(
                Clusters.dataset == hash_to_hex_decode(model_hash)
            )
        else:
            # This is a model - get clusters meeting threshold
            model_valid = select(Probabilities.cluster.label("cluster")).where(
                and_(
                    Probabilities.model == hash_to_hex_decode(model_hash),
                    Probabilities.probability >= threshold,
                )
            )

        if valid_clusters is None:
            valid_clusters = model_valid
        else:
            valid_clusters = union(valid_clusters, model_valid)

    if valid_clusters is None:
        # Handle empty lineage case
        return select(cast(null(), BYTEA).label("cluster")).where(False)

    return valid_clusters.cte("valid_clusters")


def _resolve_cluster_hierarchy(
    dataset_hash: bytes,
    model: Models,
    engine: Engine,
    threshold: float | dict[str, float] | None = None,
) -> Select:
    """
    Resolves the final cluster assignments for all records in a dataset.

    Args:
        dataset_hash: Hash of the dataset to query
        model: Model object representing the point of truth
        engine: Engine for database connection
        threshold: Optional threshold value or dict of model_name -> threshold

    Returns:
        SQLAlchemy Select statement that will resolve to (hash, id) pairs, where
        hash is the ultimate parent cluster hash and id is the original record ID
    """
    with Session(engine) as session:
        dataset_model = session.get(Models, dataset_hash)
        try:
            lineage_truths = model.get_lineage_to_dataset(model=dataset_model)
        except ValueError as e:
            raise MatchboxModelError(f"Invalid model lineage: {str(e)}") from e

        thresholds = _resolve_thresholds(
            lineage_truths=lineage_truths,
            model=model,
            threshold=threshold,
            session=session,
        )

        # Get clusters valid across all models in lineage
        valid_clusters = _union_valid_clusters(thresholds)

        # Get base mapping of IDs to clusters
        mapping_0 = (
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

        # Build recursive hierarchy CTE
        hierarchy = (
            # Base case: direct parents
            select(
                mapping_0.c.cluster_hash.label("original_cluster"),
                mapping_0.c.cluster_hash.label("child"),
                Contains.parent.label("parent"),
                literal(1).label("level"),
            )
            .select_from(mapping_0)
            .join(Contains, Contains.child == mapping_0.c.cluster_hash)
            .where(Contains.parent.in_(select(valid_clusters.c.cluster)))
            .cte("hierarchy", recursive=True)
        )

        # Recursive case
        recursive = (
            select(
                hierarchy.c.original_cluster,
                hierarchy.c.parent.label("child"),
                Contains.parent.label("parent"),
                (hierarchy.c.level + 1).label("level"),
            )
            .select_from(hierarchy)
            .join(Contains, Contains.child == hierarchy.c.parent)
            .where(Contains.parent.in_(select(valid_clusters.c.cluster)))
        )

        hierarchy = hierarchy.union_all(recursive)

        # Get highest parents
        highest_parents = (
            select(
                hierarchy.c.original_cluster,
                hierarchy.c.parent.label("highest_parent"),
                hierarchy.c.level,
            )
            .distinct(hierarchy.c.original_cluster)
            .order_by(hierarchy.c.original_cluster, hierarchy.c.level.desc())
            .cte("highest_parents")
        )

        # Final mapping with coalesced results
        final_mapping = (
            select(
                mapping_0.c.id,
                func.coalesce(
                    highest_parents.c.highest_parent, mapping_0.c.cluster_hash
                ).label("final_parent"),
            )
            .select_from(mapping_0)
            .join(
                highest_parents,
                highest_parents.c.original_cluster == mapping_0.c.cluster_hash,
                isouter=True,
            )
            .cte("final_mapping")
        )

        # Final select statement
        return select(final_mapping.c.final_parent.label("hash"), final_mapping.c.id)


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
        with the hash key of each row in Matchbox, in the requested return type
    """
    tables: list[pa.Table] = []

    if limit:
        limit_base = limit // len(selector)
        limit_remainder = limit % len(selector)

    with Session(engine) as session:
        # If a model was specified, validate and retrieve it
        truth_model = None
        if model is not None:
            truth_model = session.query(Models).filter(Models.name == model).first()
            if truth_model is None:
                raise MatchboxModelError(f"Model {model} not found")

        # Process each source dataset
        for source, fields in selector.items():
            # Get the dataset model
            dataset = (
                session.query(Models)
                .join(Sources, Sources.model == Models.hash)
                .filter(
                    Sources.schema == source.db_schema,
                    Sources.table == source.db_table,
                    Sources.id == source.db_pk,
                )
                .first()
            )

            if dataset is None:
                raise MatchboxDatasetError(
                    db_schema=source.db_schema, db_table=source.db_table
                )

            hash_query = _resolve_cluster_hierarchy(
                dataset_hash=dataset.hash,
                model=truth_model if truth_model else dataset,
                threshold=threshold,
                engine=engine,
            )

            # Apply limit if specified
            if limit:
                remain = 1 if limit_remainder > 0 else 0
                if remain:
                    limit_remainder -= 1
                hash_query = hash_query.limit(limit_base + remain)

            # Get cluster assignments
            mb_hashes = sql_to_df(hash_query, engine, return_type="arrow")

            # Get source data
            raw_data = source.to_arrow(
                fields=set([source.db_pk] + fields), pks=mb_hashes["id"].to_pylist()
            )

            # Join and select columns
            joined_table = raw_data.join(
                right_table=mb_hashes,
                keys=key_to_sqlalchemy_label(source.db_pk, source),
                right_keys="id",
                join_type="inner",
            )

            keep_cols = ["hash"] + [key_to_sqlalchemy_label(f, source) for f in fields]
            match_cols = [col for col in joined_table.column_names if col in keep_cols]

            tables.append(joined_table.select(match_cols))

    # Combine results
    result = pa.concat_tables(tables, promote_options="default")

    # Return in requested format
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
        raise ValueError(f"return_type of {return_type} not valid")


def match(
    source_id: str,
    source: str,
    target: str | list[str],
    model: str,
    engine: Engine,
    threshold: float | dict[str, float] | None = None,
) -> Match | list[Match]:
    """Matches an ID in a source dataset and returns the keys in the targets.

    To accomplish this, the function:

    * Reconstructs the model lineage from the specified model
    * Iterates through each target, and
        * Retrieves its cluster hash according to the model
        * Retrieves all other IDs in the cluster in the source dataset
        * Retrieves all other IDs in the cluster in the target dataset
    * Returns the results as Match objects, one per target
    """
    # Split source and target into schema/table
    source_schema, source_table = get_schema_table_names(source, validate=True)
    targets = [target] if isinstance(target, str) else target
    target_pairs = [get_schema_table_names(t, validate=True) for t in targets]

    with Session(engine) as session:
        # Get truth model
        truth_model = session.query(Models).filter(Models.name == model).first()
        if truth_model is None:
            raise MatchboxModelError(f"Model {model} not found")

        # Get source dataset
        source_dataset = (
            session.query(Models)
            .join(Sources, Sources.model == Models.hash)
            .filter(
                Sources.schema == source_schema,
                Sources.table == source_table,
            )
            .first()
        )
        if source_dataset is None:
            raise MatchboxDatasetError(
                db_schema=source_schema,
                db_table=source_table,
            )

        # Get target datasets
        target_datasets = []
        for schema, table in target_pairs:
            dataset = (
                session.query(Models)
                .join(Sources, Sources.model == Models.hash)
                .filter(
                    Sources.schema == schema,
                    Sources.table == table,
                )
                .first()
            )
            if dataset is None:
                raise MatchboxDatasetError(db_schema=schema, db_table=table)
            target_datasets.append(dataset)

        # Get model lineage and resolve thresholds
        lineage_truths = truth_model.get_lineage()
        thresholds = _resolve_thresholds(
            lineage_truths=lineage_truths,
            model=truth_model,
            threshold=threshold,
            session=session,
        )

        # Get valid clusters across all models
        valid_clusters = _union_valid_clusters(thresholds)

        # Unnest cluster IDs
        unnested_clusters = (
            select(
                Clusters.hash, Clusters.dataset, func.unnest(Clusters.id).label("id")
            )
            .select_from(Clusters)
            .cte("unnested_clusters")
        )

        # Find source ID's initial cluster
        source_cluster = (
            select(unnested_clusters.c.hash)
            .select_from(unnested_clusters)
            .where(
                and_(
                    unnested_clusters.c.dataset
                    == hash_to_hex_decode(source_dataset.hash),
                    unnested_clusters.c.id == source_id,
                )
            )
            .scalar_subquery()
        )

        # Build recursive hierarchy CTE going up
        hierarchy_up = (
            # Base case: direct parents
            select(
                source_cluster.label("original_cluster"),
                source_cluster.label("child"),
                Contains.parent.label("parent"),
                literal(1).label("level"),
            )
            .join(Contains, Contains.child == source_cluster)
            .where(Contains.parent.in_(select(valid_clusters.c.cluster)))
            .cte("hierarchy_up", recursive=True)
        )

        # Recursive case going up
        recursive_up = (
            select(
                hierarchy_up.c.original_cluster,
                hierarchy_up.c.parent.label("child"),
                Contains.parent.label("parent"),
                (hierarchy_up.c.level + 1).label("level"),
            )
            .select_from(hierarchy_up)
            .join(Contains, Contains.child == hierarchy_up.c.parent)
            .where(Contains.parent.in_(select(valid_clusters.c.cluster)))
        )

        hierarchy_up = hierarchy_up.union_all(recursive_up)

        # Get highest parent
        highest_parent = (
            select(hierarchy_up.c.parent)
            .order_by(hierarchy_up.c.level.desc())
            .limit(1)
            .scalar_subquery()
        )

        # Build recursive hierarchy CTE going down
        hierarchy_down = (
            # Base case: direct children from highest parent
            select(
                highest_parent.label("parent"),
                Contains.child.label("child"),
                literal(1).label("level"),
                unnested_clusters.c.dataset.label("dataset"),
                unnested_clusters.c.id.label("id"),
            )
            .select_from(Contains)
            .join(unnested_clusters, unnested_clusters.c.hash == Contains.child)
            .where(Contains.parent == highest_parent)
            .cte("hierarchy_down", recursive=True)
        )

        # Recursive case going down
        recursive_down = (
            select(
                hierarchy_down.c.parent,
                Contains.child.label("child"),
                (hierarchy_down.c.level + 1).label("level"),
                unnested_clusters.c.dataset.label("dataset"),
                unnested_clusters.c.id.label("id"),
            )
            .select_from(hierarchy_down)
            .join(Contains, Contains.parent == hierarchy_down.c.child)
            .join(unnested_clusters, unnested_clusters.c.hash == Contains.child)
        )

        hierarchy_down = hierarchy_down.union_all(recursive_down)

        # Get all matched IDs
        final_stmt = (
            select(
                hierarchy_down.c.dataset,
                hierarchy_down.c.id,
            )
            .distinct()
            .select_from(hierarchy_down)
        )
        matches = session.execute(final_stmt).all()

        # Group matches by dataset
        matches_by_dataset = {}
        for dataset_hash, id in matches:
            if dataset_hash not in matches_by_dataset:
                matches_by_dataset[dataset_hash] = set()
            matches_by_dataset[dataset_hash].add(id)

        # Create Match objects for each target
        result = []
        for target_dataset in target_datasets:
            # Get source/target table names
            source_name = f"{source_schema}.{source_table}"
            target_schema, target_table = next(
                (schema, table)
                for schema, table in target_pairs
                if session.get(Sources, target_dataset.hash).schema == schema
                and session.get(Sources, target_dataset.hash).table == table
            )
            target_name = f"{target_schema}.{target_table}"

            highest_cluster = highest_parent.scalar() if matches else None

            # Get source and target IDs
            source_ids = {
                id
                for dataset_hash, id in matches
                if dataset_hash == source_dataset.hash
            }
            target_ids = {
                id
                for dataset_hash, id in matches
                if dataset_hash == target_dataset.hash
            }

            match_obj = Match(
                cluster=highest_cluster,
                source=source_name,
                source_id=source_ids,
                target=target_name,
                target_id=target_ids,
            )
            result.append(match_obj)

        return result[0] if isinstance(target, str) else result
