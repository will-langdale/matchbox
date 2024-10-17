import io
from typing import Literal

import pandas as pd
from sqlalchemy import (
    LABEL_STYLE_TABLENAME_PLUS_COL,
    Engine,
    String,
    and_,
    func,
    or_,
    select,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.selectable import Select

from matchbox.helpers.selector import get_schema_table_names, string_to_table
from matchbox.server.postgresql.clusters import Clusters, clusters_association
from matchbox.server.postgresql.data import SourceData, SourceDataset
from matchbox.server.postgresql.dedupe import DDupeContains
from matchbox.server.postgresql.link import LinkContains
from matchbox.server.postgresql.models import Models


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
    Takes the string name of a model and returns a tuple of its SHA-1,
    and the SHA-1 list of its children.

    See query() for an overview and glossary of this function's role.

    Args:
        model_name (str): the name of a model
        engine: the SQLAlchemy engine to use

    Returns:
        A tuple of the model's SHA-1 value, and a list of the SHA-1s values
        of its decendents
    """

    with Session(engine) as session:
        model = session.query(Models).filter_by(name=model_name).first()
        model_children = get_all_children(model)
        model_children.pop(0)  # includes original model

    return model.sha1, [m.sha1 for m in model_children]


def _tree_to_reachable_stmt(model_tree: list[bytes]) -> Select:
    """
    Takes a list of models and returns a query to select their reachable
    edges.

    See query() for an overview and glossary of this function's role.

    Args:
        model_tree: a list of model SHA-1 values, likely produced by
            _parent_to_tree

    Returns:
        A SQL query in the form of a SQLAlchemy Select object
    """
    c1 = aliased(Clusters)
    c2 = aliased(Clusters)

    dd_stmt = (
        select(DDupeContains.parent, DDupeContains.child)
        .join(c1, DDupeContains.parent == c1.sha1)
        .join(clusters_association, clusters_association.c.child == c1.sha1)
        .join(Models, clusters_association.c.parent == Models.sha1)
        .where(Models.sha1.in_(model_tree))
    )

    lk_stmt = (
        select(LinkContains.parent, LinkContains.child)
        .join(c1, LinkContains.parent == c1.sha1)
        .join(c2, LinkContains.child == c2.sha1)
        .join(clusters_association, clusters_association.c.child == c1.sha1)
        .join(Models, clusters_association.c.parent == Models.sha1)
        .where(Models.sha1.in_(model_tree))
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
        parent_hash: the SHA-1 to use as the ultimate parent model, the point
            of truth

    Returns:
        A SQL query in the form of a SQLAlchemy Select object
    """
    allowed = reachable_stmt.cte("allowed")

    root = (
        select(allowed.c.parent, allowed.c.child)
        .join(Clusters, Clusters.sha1 == allowed.c.parent)
        .join(clusters_association, clusters_association.c.child == Clusters.sha1)
        .join(Models, clusters_association.c.parent == Models.sha1)
        .where(Models.sha1 == parent_hash)
        .cte("root")
    )

    recurse_top = select(root.c.parent, root.c.child).cte("recurse", recursive=True)
    recurse_bottom = select(recurse_top.c.parent, allowed.c.child).join(
        recurse_top, allowed.c.parent == recurse_top.c.child
    )
    recurse = recurse_top.union(recurse_bottom)

    return recurse


def _selector_to_data(
    selector: dict[str, list[str]],
    engine: Engine,
) -> Select:
    """
    Takes a dictionary of tables and fields, usually outputted by selectors,
    and returns a SQL statement to return them in the form of a SQLAlchemy
    Select object.

    See query() for an overview and glossary of this function's role.

    Args:
        selector: a dictionary with keys of table names, and values of a list
            of data required from that table, likely output by selector() or
            selectors()
        engine: the SQLAlchemy engine to use

    Returns:
        A SQL query in the form of a SQLAlchemy Select object
    """
    select_stmt = []
    join_stmts = []
    where_stmts = []
    for schema_table, fields in selector.items():
        db_schema, db_table = get_schema_table_names(schema_table)

        with Session(engine) as session:
            mb_dataset = (
                session.query(SourceDataset)
                .filter_by(db_schema=db_schema, db_table=db_table)
                .first()
            )

        db_table = string_to_table(db_schema, db_table, engine=engine)

        # To handle array column
        source_data_unested = select(
            SourceData.sha1, func.unnest(SourceData.id).label("id"), SourceData.dataset
        ).cte("source_data_unnested")

        select_stmt.append(db_table.c[tuple(fields)])
        join_stmts.append(
            {
                "target": db_table,
                "onclause": and_(
                    source_data_unested.c.id
                    == db_table.c[mb_dataset.db_id].cast(String),
                    source_data_unested.c.dataset == mb_dataset.uuid,
                ),
            }
        )
        where_stmts.append(db_table.c[mb_dataset.db_id] != None)  # NoQA E711

    stmt = select(
        source_data_unested.c.sha1.label("data_hash"), *select_stmt
    ).select_from(source_data_unested)

    for join_stmt in join_stmts:
        stmt = stmt.join(**join_stmt, isouter=True)

    stmt = stmt.where(or_(*where_stmts))
    stmt = stmt.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)

    return stmt


def _selector_to_pandas_dtypes(
    selector: dict[str, list[str]],
    engine: Engine,
) -> dict[str, str]:
    """
    Takes a dictionary of tables and fields, usually outputted by selectors,
    and returns a dictionary of the column names and pandas datatypes.

    Args:
        selector: a dictionary with keys of table names, and values of a list
            of data required from that table, likely output by selector() or
            selectors()
        engine: the SQLAlchemy engine to use

    Returns:
        A dictionary of column names and datatypes
    """
    types_dict = {}

    for schema_table, fields in selector.items():
        db_schema, db_table = get_schema_table_names(schema_table)
        db_table = string_to_table(db_schema, db_table, engine=engine)
        stmt = (
            select(db_table.c[tuple(fields)])
            .limit(1)
            .set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        )
        with Session(engine) as session:
            res = pd.read_sql(stmt, session.bind).convert_dtypes(
                dtype_backend="pyarrow"
            )
        types_dict = types_dict | res.dtypes.apply(lambda x: x.name).to_dict()

    return types_dict


def query(
    selector: dict[str, list[str]],
    engine: Engine,
    return_type: Literal["pandas", "sqlalchemy"] = "pandas",
    model: str | None = None,
    limit: int = None,
) -> pd.DataFrame | ChunkedIteratorResult:
    """
    Takes the dictionaries of tables and fields outputted by selectors and
    queries database for them. If a model "point of truth" is supplied, will
    attach the clusters this data belongs to.

    To resolve a model point of truth's clusters with the data that belongs to
    them we:

        * Find all the model's decendent models: the "model tree"
        * Filter all clusters in the system that models in this tree create:
        the "reachable clusters"
        * Union the LinkContains and DDupeContains tables and filter to rows
        that connect reachable clustes: the "reachable edges"
        * Recurse on reachable edges to create a lookup of the point of
        truth's cluster SHA-1 to the ultimate decendent SHA-1 in the chain:
        the SHA-1 key of the SourceData

    Args:
        selector: a dictionary with keys of table names, and values of a list
            of data required from that table, likely output by selector() or
            selectors()
        model (str): Optional. A model considered the point of truth to decide which
            clusters data belongs to
        return_type (str): the form to return data in, one of "pandas" or
            "sqlalchemy"
        engine: the SQLAlchemy engine to use
        limit (int): the number to use in a limit clause. Useful for testing

    Returns:
        A table containing the requested data from each table, unioned together,
        with the SHA-1 key of each row in the Company Matching Framework, in the
        requested return type. If a model was given, also contains the SHA-1
        cluster of each data -- the company entity the data belongs to according
        to the model.
    """
    if model is None:
        # We want raw data with no clusters
        final_stmt = _selector_to_data(selector, engine=engine)
    else:
        # We want raw data with clusters attached
        parent, child = _parent_to_tree(model, engine=engine)
        if len(parent) == 0:
            raise ValueError(f"Model {model} not found")
        tree = [parent] + child
        reachable_stmt = _tree_to_reachable_stmt(tree)
        lookup_stmt = _reachable_to_parent_data_stmt(reachable_stmt, parent)
        data_stmt = _selector_to_data(selector, engine=engine).cte()

        final_stmt = select(lookup_stmt.c.parent.label("cluster_hash"), data_stmt).join(
            lookup_stmt, lookup_stmt.c.child == data_stmt.c.data_hash
        )

    if limit is not None:
        final_stmt = final_stmt.limit(limit)

    if return_type == "pandas":
        # Detect datatypes
        selector_dtypes = _selector_to_pandas_dtypes(selector, engine=engine)
        default_dtypes = {
            "cluster_hash": "string[pyarrow]",
            "data_hash": "string[pyarrow]",
        }

        with engine.connect() as conn:
            # Compile query
            cursor = conn.connection.cursor()
            compiled = final_stmt.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"render_postcompile": True},
            )
            compiled_bound = cursor.mogrify(str(compiled), compiled.params)
            sql = compiled_bound.decode("utf-8")
            copy_sql = f"copy ({sql}) to stdout with csv header"

            # Load from Postgres to memory
            store = io.StringIO()
            cursor.copy_expert(copy_sql, store)
            store.seek(0)

            # Read to pandas
            res = pd.read_csv(
                store, dtype=default_dtypes | selector_dtypes, engine="pyarrow"
            ).convert_dtypes(dtype_backend="pyarrow")

            # Manually convert SHA-1s to bytes correctly
            if "data_hash" in res.columns:
                res.data_hash = res.data_hash.str[2:].apply(bytes.fromhex)
                res.data_hash = res.data_hash.astype("binary[pyarrow]")
            if "cluster_hash" in res.columns:
                res.cluster_hash = res.cluster_hash.str[2:].apply(bytes.fromhex)
                res.cluster_hash = res.cluster_hash.astype("binary[pyarrow]")

    elif return_type == "sqlalchemy":
        with Session(engine) as session:
            res = session.execute(final_stmt)
    else:
        ValueError(f"return_type of {return_type} not valid")

    return res
