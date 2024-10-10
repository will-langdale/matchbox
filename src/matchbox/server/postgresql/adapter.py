import io
from typing import Literal

import pandas as pd
from sqlalchemy import (
    select,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.orm import Session

from matchbox.server.base import MatchboxDBAdapter, MatchboxModelAdapter
from matchbox.server.postgresql import (
    ENGINE,
    Clusters,
    DDupeProbabilities,
    Dedupes,
    LinkProbabilities,
    Links,
    Models,
    ModelsFrom,
    SourceData,
    SourceDataset,
    clusters_association,
)
from matchbox.server.postgresql.utils.db import get_model_subgraph
from matchbox.server.postgresql.utils.selector import (
    _parent_to_tree,
    _reachable_to_parent_data_stmt,
    _selector_to_data,
    _selector_to_pandas_dtypes,
    _tree_to_reachable_stmt,
    get_all_parents,
)


class MergesUnion:
    """A thin wrapper around Dedupes and Links to provide a count method."""

    def count(self) -> int:
        return Dedupes.count() + Links.count()


class ProposesUnion:
    """A thin wrapper around probability classes to provide a count method."""

    def count(self) -> int:
        return DDupeProbabilities.count() + LinkProbabilities.count()


class MatchboxPostgres(MatchboxDBAdapter):
    """A PostgreSQL adapter for Matchbox."""

    engine = ENGINE

    def __init__(self):
        self.datasets = SourceDataset
        self.models = Models
        self.models_from = ModelsFrom
        self.data = SourceData
        self.clusters = Clusters
        self.creates = clusters_association
        self.merges = MergesUnion()
        self.proposes = ProposesUnion()

    def query(
        self,
        selector: dict[str, list[str]],
        model: str | None = None,
        return_type: Literal["pandas", "sqlalchemy"] | None = None,
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
            final_stmt = _selector_to_data(selector, engine=self.engine)
        else:
            # We want raw data with clusters attached
            parent, child = _parent_to_tree(model, engine=self.engine)
            if len(parent) == 0:
                raise ValueError(f"Model {model} not found")
            tree = [parent] + child
            reachable_stmt = _tree_to_reachable_stmt(tree)
            lookup_stmt = _reachable_to_parent_data_stmt(reachable_stmt, parent)
            data_stmt = _selector_to_data(selector, engine=self.engine).cte()

            final_stmt = select(
                lookup_stmt.c.parent.label("cluster_sha1"), data_stmt
            ).join(lookup_stmt, lookup_stmt.c.child == data_stmt.c.data_sha1)

        if limit is not None:
            final_stmt = final_stmt.limit(limit)

        if return_type == "pandas":
            # Detect datatypes
            selector_dtypes = _selector_to_pandas_dtypes(selector, engine=self.engine)
            default_dtypes = {
                "cluster_sha1": "string[pyarrow]",
                "data_sha1": "string[pyarrow]",
            }

            with self.engine.connect() as conn:
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
                if "data_sha1" in res.columns:
                    res.data_sha1 = res.data_sha1.str[2:].apply(bytes.fromhex)
                    res.data_sha1 = res.data_sha1.astype("binary[pyarrow]")
                if "cluster_sha1" in res.columns:
                    res.cluster_sha1 = res.cluster_sha1.str[2:].apply(bytes.fromhex)
                    res.cluster_sha1 = res.cluster_sha1.astype("binary[pyarrow]")

        elif return_type == "sqlalchemy":
            with Session(self.engine) as session:
                res = session.execute(final_stmt)
        else:
            ValueError(f"return_type of {return_type} not valid")

        return res

    def index(self, db_schema: str, db_table: str) -> None:
        # logic moved from cmf.admin.add_dataset()
        pass

    def get_model_subgraph(self) -> dict:
        """Get the full subgraph of a model."""
        return get_model_subgraph(engine=self.engine)

    def get_model(self, model: str) -> MatchboxModelAdapter:
        # logic moved from a million different places in unit tests and results
        pass

    def delete_model(self, model: str, certain: bool = False) -> None:
        """
        Deletes:

        * The model from the model table
        * The model's edges to its child models from the models_from table
        * The creates edges the model made from the clusters_association table
        * Any probability values associated with the model from the ddupe_probabilities
            and link_probabilities tables
        * All of the above for all parent models. As every model is defined by
            its children, deleting a model means cascading deletion to all ancestors

        It DOESN'T delete the raw clusters or probability nodes from the ddupes and
            links tables, which retain any validation attached to them.
        """
        with Session(self.engine) as session:
            target_model = session.query(Models).filter_by(name=model).first()
            all_parents = get_all_parents(target_model)
            if certain:
                for m in all_parents:
                    session.delete(m)
                session.commit()
            else:
                raise ValueError(
                    "This operation will delete the models "
                    f"{', '.join([m.name for m in all_parents])}, as well as all "
                    "references to clusters and probabilities they have created."
                    "\n\n"
                    "It will not delete validation associated with these "
                    "clusters or probabilities."
                    "\n\n"
                    "If you're sure you want to continue, rerun with certain=True"
                )

    def insert_model(self, model: str) -> None:
        # logic moved from cmf.data.results.ResultsBaseDataclass._model_to_cmf()
        pass
