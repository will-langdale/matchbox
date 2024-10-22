from typing import Literal, TypeVar

import pyarrow as pa
from pandas import ArrowDtype, DataFrame
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.exceptions import MatchboxModelError
from matchbox.server.models import Source
from matchbox.server.postgresql.clusters import Clusters, Creates
from matchbox.server.postgresql.data import SourceData, SourceDataset
from matchbox.server.postgresql.dedupe import DDupeContains
from matchbox.server.postgresql.link import LinkContains
from matchbox.server.postgresql.models import Models

T = TypeVar("T")


def hash_to_hex_decode(hash: bytes) -> bytes:
    """A workround for PostgreSQL so we can compile the query and use ConnectorX."""
    return func.decode(hash.hex(), "hex")


def key_to_sqlalchemy_label(key: str, source: Source) -> str:
    """Converts a key to the SQLAlchemy LABEL_STYLE_TABLENAME_PLUS_COL."""
    return f"{source.db_schema}_{source.db_table}_{key}"


def get_all_parents(model: Models | list[Models]) -> list[Models]:
    """
    Takes a Models object and returns all items in its parent tree.

    Intended as a lower-level function for other functions to use.

    Args:
        model: a single Model object.

    Returns:
        A list of all the model's ancestor models.
    """
    result = []
    if isinstance(model, list):
        for mod in model:
            parents = get_all_parents(mod)
            result.append(parents)
        return result
    elif isinstance(model, Models):
        result.append(model)
        parent_neighbours = model.parent_neighbours()
        if len(parent_neighbours) == 0:
            return result
        else:
            for mod in parent_neighbours:
                parents = get_all_parents(mod)
                result += parents
            return result


def get_all_children(model: Models | list[Models]) -> list[Models]:
    """
    Takes a Models object and returns all items in its child tree.

    Intended as a lower-level function for other functions to use.

    Args:
        model: a single Model object.

    Returns:
        A list of all the model's decendent models.
    """
    result = []
    if isinstance(model, list):
        for mod in model:
            children = get_all_children(mod)
            result.append(children)
        return result
    elif isinstance(model, Models):
        result.append(model)
        child_neighbours = model.child_neighbours()
        if len(child_neighbours) == 0:
            return result
        else:
            for mod in child_neighbours:
                children = get_all_children(mod)
                result += children
            return result


def _parent_to_tree(model_name: str, engine: Engine) -> tuple[bytes, list[bytes]]:
    """
    Takes the string name of a model and returns a tuple of its hash,
    and the hash list of its children.

    See query() for an overview and glossary of this function's role.

    Args:
        model_name (str): the name of a model
        engine: the SQLAlchemy engine to use

    Returns:
        A tuple of the model's hash value, and a list of the hashs values
        of its decendents
    """

    with Session(engine) as session:
        if model := session.query(Models).filter_by(name=model_name).first():
            model_children = get_all_children(model)
            model_children.pop(0)  # includes original model
        else:
            raise MatchboxModelError(model_name=model_name)

    return model.sha1, [m.sha1 for m in model_children]


def _tree_to_reachable_stmt(model_tree: list[bytes]) -> Select:
    """
    Takes a list of models and returns a query to select their reachable
    edges.

    See query() for an overview and glossary of this function's role.

    Args:
        model_tree: a list of model hash values, likely produced by
            _parent_to_tree

    Returns:
        A SQL query in the form of a SQLAlchemy Select object
    """
    c1 = aliased(Clusters)
    c2 = aliased(Clusters)

    bytea_array = array([hash_to_hex_decode(m) for m in model_tree])
    subquery = select(func.unnest(bytea_array).label("hash")).subquery()

    dd_stmt = (
        select(DDupeContains.parent, DDupeContains.child)
        .join(c1, DDupeContains.parent == c1.sha1)
        .join(Creates, Creates.child == c1.sha1)
        .join(Models, Creates.parent == Models.sha1)
        .where(Models.sha1.in_(select(subquery.c.hash)))
    )

    lk_stmt = (
        select(LinkContains.parent, LinkContains.child)
        .join(c1, LinkContains.parent == c1.sha1)
        .join(c2, LinkContains.child == c2.sha1)
        .join(Creates, Creates.child == c1.sha1)
        .join(Models, Creates.parent == Models.sha1)
        .where(Models.sha1.in_(select(subquery.c.hash)))
    )

    return dd_stmt.union(lk_stmt)


def _reachable_to_parent_data_stmt(
    reachable_stmt: Select, parent_hash: bytes
) -> Select:
    """
    Takes a select statement representing the reachable edges of a parent
    model and returns a statement to create a parent cluster to child data
    lookup

    See query() for an overview and glossary of this function's role.

    Args:
        reachable_stmt: a SQLAlchemy Select object that defines the reachable
            edges of the combined LinkContains and DDupeContains tables
        parent_hash: the hash to use as the ultimate parent model, the point
            of truth

    Returns:
        A SQL query in the form of a SQLAlchemy Select object
    """
    allowed = reachable_stmt.cte("allowed")

    root = (
        select(allowed.c.parent, allowed.c.child)
        .join(Clusters, Clusters.sha1 == allowed.c.parent)
        .join(Creates, Creates.child == Clusters.sha1)
        .join(Models, Creates.parent == Models.sha1)
        .where(Models.sha1 == hash_to_hex_decode(parent_hash))
        .cte("root")
    )

    recurse_top = select(root.c.parent, root.c.child).cte("recurse", recursive=True)
    recurse_bottom = select(recurse_top.c.parent, allowed.c.child).join(
        recurse_top, allowed.c.parent == recurse_top.c.child
    )
    recurse = recurse_top.union(recurse_bottom)

    return recurse


def _souce_to_hashes(
    source: Source,
    engine: Engine,
) -> Select:
    """
    Takes a single selector and returns a SQL statement to return its data.

    See query() for an overview and glossary of this function's role.

    Args:
        selector: a dictionary with keys of table names, and values of a list
            of data required from that table, likely output by selector() or
            selectors()
        engine: the SQLAlchemy engine to use

    Returns:
        A SQL query in the form of a SQLAlchemy Select object
    """
    with Session(engine) as session:
        dataset = (
            session.query(SourceDataset)
            .filter_by(db_schema=source.db_schema, db_table=source.db_table)
            .first()
        )
        dataset_uuid = dataset.uuid

    stmt = select(
        SourceData.sha1.label("data_hash"),
        func.unnest(SourceData.id).label("id"),  # Handle array column
    ).where(SourceData.dataset == dataset_uuid)

    return stmt


def _model_to_hashes(
    source: Source,
    model: str,
    engine: Engine,
) -> Select:
    """Takes a source and model and returns a statement to retrieve its data.

    This is the composite function that traverses the tree and returns all the
    relevant data for a model's point of truth. The statement it returns will
    give the data hash, the cluster hash, and the primary key of the data for
    this source, according to the model.

    To resolve a model point of truth's clusters with the data that belongs to
    them we:

        * Find all the model's decendent models: the "model tree"
        * Filter all clusters in the system that models in this tree create:
        the "reachable clusters"
        * Union the LinkContains and DDupeContains tables and filter to rows
        that connect reachable clustes: the "reachable edges"
        * Recurse on reachable edges to create a lookup of the point of
        truth's cluster hash to the ultimate decendent hash in the chain:
        the hash key of the SourceData
    """
    parent, child = _parent_to_tree(model, engine=engine)
    if not parent:
        raise ValueError(f"Model {model} not found")
    tree = [parent] + child
    reachable_stmt = _tree_to_reachable_stmt(tree)
    lookup_stmt = _reachable_to_parent_data_stmt(reachable_stmt, parent)
    data_stmt = _souce_to_hashes(source, engine=engine).cte()

    final_stmt = select(lookup_stmt.c.parent.label("cluster_hash"), data_stmt).join(
        lookup_stmt, lookup_stmt.c.child == data_stmt.c.data_hash
    )

    return final_stmt


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
        if model is None:
            # We want raw data with no clusters
            hash_query = _souce_to_hashes(source, engine=engine)
        else:
            # We want raw data with clusters attached
            hash_query = _model_to_hashes(source, model, engine=engine)

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
        keep_cols = ["cluster_hash", "data_hash"] + [
            key_to_sqlalchemy_label(f, source) for f in fields
        ]
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
