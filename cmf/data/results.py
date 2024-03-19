import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import rustworkx as rx
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame, concat
from pg_bulk_ingest import Delete, Upsert, ingest
from pydantic import BaseModel, ConfigDict, computed_field, model_validator
from sqlalchemy import (
    Engine,
    Table,
    bindparam,
    delete,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from cmf.data import utils as du
from cmf.data.clusters import Clusters, clusters_association
from cmf.data.data import SourceData
from cmf.data.db import ENGINE
from cmf.data.dedupe import DDupeContains, DDupeProbabilities, Dedupes
from cmf.data.exceptions import CMFDBDataError
from cmf.data.link import LinkContains, LinkProbabilities, Links
from cmf.data.models import Models, ModelsFrom

logic_logger = logging.getLogger("cmf_logic")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


class ResultsBaseDataclass(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: DataFrame
    run_name: str
    description: str
    left: str
    right: str

    _expected_fields: List[str]
    _batch_size: int = int(os.environ["BATCH_SIZE"])

    @model_validator(mode="after")
    def _check_dataframe(self) -> Table:
        """Verifies the table contains the expected fields."""
        table_fields = sorted(self.dataframe.columns)
        expected_fields = sorted(self._expected_fields)

        if table_fields != expected_fields:
            raise ValueError(f"Expected {expected_fields}. \n" f"Found {table_fields}.")

        return self

    @computed_field
    @property
    def metadata(self) -> str:
        return f"{self.run_name}, {self._get_results_type()}"

    @abstractmethod
    def inspect_with_source(self) -> DataFrame:
        """Enriches the results with the source data."""
        return

    @abstractmethod
    def to_df(self) -> DataFrame:
        """Returns the results as a DataFrame."""
        return

    @abstractmethod
    def _deduper_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a deduper to the CMF database."""
        return

    @abstractmethod
    def _linker_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a linker to the CMF database."""
        return

    @classmethod
    def _get_results_type(cls):
        return cls.__name__

    def _model_to_cmf(
        self, deduplicates: bytes = None, engine: Engine = ENGINE
    ) -> None:
        """Writes the model to the CMF.

        Raises
            CMFDBDataError if, for a linker, the source models weren't found in
                the database
        """
        with Session(engine) as session:
            if deduplicates is None:
                # Linker
                # Construct model SHA1 from parent model SHA1s
                left_sha1 = du.model_name_to_sha1(self.left, engine=engine)
                right_sha1 = du.model_name_to_sha1(self.right, engine=engine)

                model_sha1 = du.list_to_value_ordered_sha1(
                    [bytes(self.run_name, encoding="utf-8"), left_sha1, right_sha1]
                )
            else:
                # Deduper
                model_sha1 = du.list_to_value_ordered_sha1([self.run_name, self.left])

            model = Models(
                sha1=model_sha1,
                name=self.run_name,
                description=self.description,
                deduplicates=deduplicates,
            )

            session.merge(model)
            session.commit()

            if deduplicates is None:
                # Linker
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

    def to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results to the CMF database."""
        if self.left == self.right:
            # Deduper
            # Write model
            logic_logger.info(f"[{self.metadata}] Registering model")
            self._model_to_cmf(
                deduplicates=du.table_name_to_uuid(self.left, engine=engine),
                engine=engine,
            )

            # Write data
            if self.dataframe.shape[0] == 0:
                logic_logger.info(f"[{self.metadata}] No deduplication data to insert")
            else:
                logic_logger.info(
                    f"[{self.metadata}] Writing deduplication data "
                    f"with batch size {self._batch_size}"
                )
                self._deduper_to_cmf(engine=engine)
        else:
            # Linker
            # Write model
            logic_logger.info(f"[{self.metadata}] Registering model")
            self._model_to_cmf(engine=engine)

            # Write data
            if self.dataframe.shape[0] == 0:
                logic_logger.info(f"[{self.metadata}] No link data to insert")
            else:
                logic_logger.info(
                    f"[{self.metadata}] Writing link data "
                    f"with batch size {self._batch_size}"
                )
                self._linker_to_cmf(engine=engine)

        logic_logger.info(f"[{self.metadata}] Complete!")


class ProbabilityResults(ResultsBaseDataclass):
    """Probabilistic matches produced by linkers and dedupers.

    Inherits the following attributes from ResultsBaseDataclass.

    _expected_fields enforces the shape of the dataframe.

    Args:
        dataframe (DataFrame): the DataFrame holding the results
        run_name (str): the name of the run or experiment
        description (str): a description of the model generating the results
        left (str): the source dataset or source model for the left side of
            the comparison
        right (str): the source dataset or source model for the right side of
            the comparison
    """

    _expected_fields: List[str] = [
        "left_id",
        "right_id",
        "probability",
    ]

    def inspect_with_source(
        self, left_data: DataFrame, left_key: str, right_data: DataFrame, right_key: str
    ) -> DataFrame:
        """Enriches the results with the source data."""
        df = (
            self.to_df()
            .filter(["left_id", "right_id", "probability"])
            .assign(
                left_id=lambda d: d.left_id.apply(str),
                right_id=lambda d: d.right_id.apply(str),
            )
            .merge(
                left_data.assign(**{left_key: lambda d: d[left_key].apply(str)}),
                how="left",
                left_on="left_id",
                right_on=left_key,
            )
            .drop(columns=[left_key])
            .merge(
                right_data.assign(**{right_key: lambda d: d[right_key].apply(str)}),
                how="left",
                left_on="right_id",
                right_on=right_key,
            )
            .drop(columns=[right_key])
        )

        return df

    def to_df(self) -> DataFrame:
        """Returns the results as a DataFrame."""
        df = self.dataframe.assign(
            left=self.left, right=self.right, model=self.run_name
        ).convert_dtypes(dtype_backend="pyarrow")[
            ["model", "left", "left_id", "right", "right_id", "probability"]
        ]

        return df

    def _prep_to_cmf(self, df: DataFrame, engine: Engine = ENGINE) -> DataFrame:
        """Transform and validate data, calculate SHA-1 hash."""
        pre_prep_df = df
        cols = ["left_id", "right_id"]

        # Verify data is in the CMF
        # Check SourceData for dedupers and Clusters for linkers
        if self.left == self.right:
            # Deduper
            Source = SourceData
            tgt_col = "data_sha1"
        else:
            # Linker
            Source = Clusters
            tgt_col = "cluster_sha1"

        pre_prep_df[cols] = pre_prep_df[cols].astype("binary[pyarrow]")

        for col in cols:
            data_unique = pre_prep_df[col].unique().tolist()

            with Session(engine) as session:
                data_inner_join = (
                    session.query(Source)
                    .filter(
                        Source.sha1.in_(
                            bindparam(
                                "ins_sha1s",
                                data_unique,
                                expanding=True,
                            )
                        )
                    )
                    .all()
                )

            if len(data_inner_join) != len(data_unique):
                raise CMFDBDataError(
                    message=(
                        f"Some items in {col} don't exist the target table. "
                        f"Did you use {tgt_col} as your ID when deduplicating?"
                    ),
                    source=Source,
                )

        # Transform for insert
        pre_prep_df["sha1"] = du.columns_to_value_ordered_sha1(
            data=self.dataframe, columns=cols
        )
        pre_prep_df.sha1 = pre_prep_df.sha1.astype("binary[pyarrow]")
        pre_prep_df = pre_prep_df.rename(
            columns={"left_id": "left", "right_id": "right"}
        )

        return pre_prep_df[["sha1", "left", "right", "probability"]]

    def _deduper_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a deduper to the CMF database.

        * Turns data into a left, right, sha1, probability dataframe
        * Upserts data then queries it back as SQLAlchemy objects
        * Attaches these objects to the model

        Raises:
            CMFSourceTableError is source tables aren't in the wider database
            CMFDBDataError if current model wasn't inserted correctly
        """
        probabilities_to_add = self._prep_to_cmf(self.dataframe, engine=engine)

        logic_logger.info(
            f"[{self.metadata}] Processed %s link probabilities",
            len(probabilities_to_add),
        )

        # Validate tables exist
        _ = du.schema_table_to_table(full_name=self.left, validate=True, engine=engine)
        _ = du.schema_table_to_table(full_name=self.right, validate=True, engine=engine)

        with Session(engine) as session:
            # Add probabilities
            # Get model
            model = session.query(Models).filter_by(name=self.run_name).first()
            model_sha1 = model.sha1

            if model is None:
                raise CMFDBDataError(source=Models, data=self.run_name)

            # Clear old model probabilities
            old_ddupe_probs_subquery = (
                model.proposes_dedupes.select().with_only_columns(
                    DDupeProbabilities.model
                )
            )

            session.execute(
                delete(DDupeProbabilities).where(
                    DDupeProbabilities.model.in_(old_ddupe_probs_subquery)
                )
            )

            session.commit()

            logic_logger.info(
                f"[{self.metadata}] Removed old deduplication probabilities"
            )

        with engine.connect() as conn:
            logic_logger.info(
                f"[{self.metadata}] Inserting %s deduplication objects",
                probabilities_to_add.shape[0],
            )

            # Upsert dedupe nodes
            # Create data batching function and pass it to ingest
            fn_dedupe_batch = du.data_to_batch(
                dataframe=probabilities_to_add[list(Dedupes.__table__.columns.keys())],
                table=Dedupes.__table__,
                batch_size=self._batch_size,
            )

            ingest(
                conn=conn,
                metadata=Dedupes.metadata,
                batches=fn_dedupe_batch,
                upsert=Upsert.IF_PRIMARY_KEY,
                delete=Delete.OFF,
            )

            # Insert dedupe probabilities
            fn_dedupe_probs_batch = du.data_to_batch(
                dataframe=(
                    probabilities_to_add.assign(model=model_sha1).rename(
                        columns={"sha1": "ddupe"}
                    )[list(DDupeProbabilities.__table__.columns.keys())]
                ),
                table=DDupeProbabilities.__table__,
                batch_size=self._batch_size,
            )

            ingest(
                conn=conn,
                metadata=DDupeProbabilities.metadata,
                batches=fn_dedupe_probs_batch,
                upsert=Upsert.IF_PRIMARY_KEY,
                delete=Delete.OFF,
            )

            logic_logger.info(
                f"[{self.metadata}] Inserted all %s deduplication objects",
                probabilities_to_add.shape[0],
            )

    def _linker_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a linker to the CMF database.

        * Turns data into a left, right, sha1, probability dataframe
        * Upserts data then queries it back as SQLAlchemy objects
        * Attaches these objects to the model

        Raises:
            CMFDBDataError if current model wasn't inserted correctly
        """
        probabilities_to_add = self._prep_to_cmf(self.dataframe, engine=engine)

        logic_logger.info(
            f"[{self.metadata}] Processed %s link probabilities",
            len(probabilities_to_add),
        )

        with Session(engine) as session:
            # Add probabilities
            # Get model
            model = session.query(Models).filter_by(name=self.run_name).first()
            model_sha1 = model.sha1

            if model is None:
                raise CMFDBDataError(source=Models, data=self.run_name)

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

            logic_logger.info(f"[{self.metadata}] Removed old link probabilities")

        with engine.connect() as conn:
            logic_logger.info(
                f"[{self.metadata}] Inserting %s link objects",
                probabilities_to_add.shape[0],
            )

            # Upsert link nodes
            # Create data batching function and pass it to ingest
            fn_link_batch = du.data_to_batch(
                dataframe=probabilities_to_add[list(Links.__table__.columns.keys())],
                table=Links.__table__,
                batch_size=self._batch_size,
            )

            ingest(
                conn=conn,
                metadata=Links.metadata,
                batches=fn_link_batch,
                upsert=Upsert.IF_PRIMARY_KEY,
                delete=Delete.OFF,
            )

            # Insert link probabilities
            fn_link_probs_batch = du.data_to_batch(
                dataframe=(
                    probabilities_to_add.assign(model=model_sha1).rename(
                        columns={"sha1": "link"}
                    )[list(LinkProbabilities.__table__.columns.keys())]
                ),
                table=LinkProbabilities.__table__,
                batch_size=self._batch_size,
            )

            ingest(
                conn=conn,
                metadata=LinkProbabilities.metadata,
                batches=fn_link_probs_batch,
                upsert=Upsert.IF_PRIMARY_KEY,
                delete=Delete.OFF,
            )

            logic_logger.info(
                f"[{self.metadata}] Inserted all %s link objects",
                probabilities_to_add.shape[0],
            )


class ClusterResults(ResultsBaseDataclass):
    """Cluster data produced by using to_clusters on ProbabilityResults.

    Inherits the following attributes from ResultsBaseDataclass.

    _expected_fields enforces the shape of the dataframe.

    Args:
        dataframe (DataFrame): the DataFrame holding the results
        run_name (str): the name of the run or experiment
        description (str): a description of the model generating the results
        left (str): the source dataset or source model for the left side of
            the comparison
        right (str): the source dataset or source model for the right side of
            the comparison
    """

    _expected_fields: List[str] = ["parent", "child"]

    def inspect_with_source(
        self,
        left_data: DataFrame,
        left_key: str,
        right_data: DataFrame,
        right_key: str,
    ) -> DataFrame:
        """Enriches the results with the source data."""
        return (
            self.to_df()
            .filter(["parent", "child"])
            .map(str)
            .merge(
                left_data.assign(**{left_key: lambda d: d[left_key].apply(str)}),
                how="left",
                left_on="child",
                right_on=left_key,
            )
            .drop(columns=[left_key])
            .merge(
                right_data.assign(**{right_key: lambda d: d[right_key].apply(str)}),
                how="left",
                left_on="child",
                right_on=right_key,
            )
            .drop(columns=[right_key])
        )

    def to_df(self) -> DataFrame:
        """Returns the results as a DataFrame."""
        return self.dataframe.copy().convert_dtypes(dtype_backend="pyarrow")

    def _to_cmf_logic(
        self,
        contains_class: Union[DDupeContains, LinkContains],
        engine: Engine = ENGINE,
    ) -> None:
        """Handles common logic for writing dedupe or link clusters to the database.

        In ClusterResults, the only difference is the tables being written to.

        * Adds the new cluster nodes
        * Adds model endorsement of these nodes with "creates" edge
        * Adds the contains edges to show which clusters contain which

        Args:
            contains_class: the target table, one of DDupeContains or LinkContains
            engine: a SQLAlchemy Engine object for the database

        Raises:
            CMFDBDataError if model wasn't inserted correctly
        """
        Contains = contains_class
        with Session(engine) as session:
            # Add clusters
            # Get model
            model = session.query(Models).filter_by(name=self.run_name).first()
            model_sha1 = model.sha1

            if model is None:
                raise CMFDBDataError(source=Models, data=self.run_name)

            # Clear old model endorsements
            old_cluster_creates_subquery = model.creates.select().with_only_columns(
                Clusters.sha1
            )

            session.execute(
                delete(clusters_association).where(
                    clusters_association.c.child.in_(old_cluster_creates_subquery)
                )
            )

            session.commit()

            logic_logger.info(f"[{self.metadata}] Removed old clusters")

        with engine.connect() as conn:
            logic_logger.info(
                f"[{self.metadata}] Inserting %s cluster objects",
                self.dataframe.shape[0],
            )

            clusters_prepped = self.dataframe.astype("binary[pyarrow]")

            # Upsert cluster nodes
            # Create data batching function and pass it to ingest
            fn_cluster_batch = du.data_to_batch(
                dataframe=(
                    clusters_prepped.drop_duplicates(subset="parent").rename(
                        columns={"parent": "sha1"}
                    )[list(Clusters.__table__.columns.keys())]
                ),
                table=Clusters.__table__,
                batch_size=self._batch_size,
            )

            ingest(
                conn=conn,
                metadata=Clusters.metadata,
                batches=fn_cluster_batch,
                upsert=Upsert.IF_PRIMARY_KEY,
                delete=Delete.OFF,
            )

            # Insert cluster contains
            fn_cluster_contains_batch = du.data_to_batch(
                dataframe=clusters_prepped[list(Contains.__table__.columns.keys())],
                table=Contains.__table__,
                batch_size=self._batch_size,
            )

            ingest(
                conn=conn,
                metadata=Contains.metadata,
                batches=fn_cluster_contains_batch,
                upsert=Upsert.IF_PRIMARY_KEY,
                delete=Delete.OFF,
            )

            # Insert cluster proposed by
            fn_cluster_proposed_batch = du.data_to_batch(
                dataframe=(
                    clusters_prepped.drop("child", axis=1)
                    .rename(columns={"parent": "child"})
                    .assign(parent=model_sha1)[
                        list(clusters_association.columns.keys())
                    ]
                ),
                table=clusters_association,
                batch_size=self._batch_size,
            )

            ingest(
                conn=conn,
                metadata=clusters_association.metadata,
                batches=fn_cluster_proposed_batch,
                upsert=Upsert.IF_PRIMARY_KEY,
                delete=Delete.OFF,
            )

            logic_logger.info(
                f"[{self.metadata}] Inserted all %s cluster objects",
                self.dataframe.shape[0],
            )

    def _deduper_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a deduper to the CMF database."""
        self._to_cmf_logic(contains_class=DDupeContains, engine=engine)

    def _linker_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a linker to the CMF database."""
        self._to_cmf_logic(contains_class=LinkContains, engine=engine)


def get_unclustered(
    clusters: ClusterResults, data: DataFrame, key: str
) -> ClusterResults:
    """
    Creates a ClusterResult for data that wasn't linked or deduped.

    When writing to the Company Matching Framework this allows a model to
    endorse an existing Cluster if it wasn't linked or deduped.

    Args:
        clusters (ClusterResults): a ClusterResults generated by a linker or deduper
        data (DataFrame): cleaned data that went into the model
        key (str): the column that was matched, usually data_sha1 or cluster_sha1

    Returns:
        A ClusterResults object
    """
    no_parent = {"parent": [], "child": []}

    clustered_children = set(clusters.to_df().child)
    unclustered_children = set(data[key].map(bytes))

    cluster_diff = list(unclustered_children.difference(clustered_children))

    no_parent = {
        "parent": cluster_diff,
        "child": cluster_diff,
    }

    return ClusterResults(
        dataframe=DataFrame(no_parent).convert_dtypes(dtype_backend="pyarrow"),
        run_name=clusters.run_name,
        description=clusters.description,
        left=clusters.left,
        right=clusters.right,
    )


def to_clusters(
    *data: Optional[DataFrame],
    results: ProbabilityResults,
    key: str,
    threshold: float = 0.0,
) -> ClusterResults:
    """
    Takes a models probabilistic outputs and turns them into clusters.

    If the original data is supplied, will add unmatched data, the expected
    output for adding to the database.

    Args:
        results (ProbabilityResults): an object of class ProbabilityResults
        key (str): the column that was matched, usually data_sha1 or cluster_sha1
        threshold (float): the value above which to consider probabilities true
        data (DataFrame): (optional) Any number of cleaned data that went into
            the model. Typically this is one dataset for a deduper or two for a
            linker
    Returns
        A ClusterResults object
    """
    all_edges = (
        results.dataframe.query("probability >= @threshold")
        .filter(["left_id", "right_id"])
        .astype("binary[pyarrow]")
        .itertuples(index=False, name=None)  # generator saves on memory
    )

    G = rx.PyGraph()
    added = {}

    for edge in all_edges:
        edge_idx = []
        for sha1 in edge:
            sha1_idx = added.get(sha1)
            if sha1_idx is None:
                sha1_idx = G.add_node(sha1)
                added[sha1] = sha1_idx
            edge_idx.append(sha1_idx)
        edge_idx.append(None)
        _ = G.add_edge(*edge_idx)

    res = {"parent": [], "child": []}  # new clusters, existing hashes

    for component in rx.connected_components(G):
        child_hashes = []
        for child in component:
            child_hash = G.get_node_data(child)
            child_hashes.append(child_hash)
            res["child"].append(child_hash)

        # Must be sorted to be symmetric
        parent_hash = du.list_to_value_ordered_sha1(child_hashes)

        res["parent"] += [parent_hash] * len(component)

    matched_results = ClusterResults(
        dataframe=DataFrame(res).convert_dtypes(dtype_backend="pyarrow"),
        run_name=results.run_name,
        description=results.description,
        left=results.left,
        right=results.right,
    )

    if len(data) > 0:
        all_unmatched_results = []

        for df in data:
            unmatched_results = get_unclustered(
                clusters=matched_results, data=df, key=key
            )
            all_unmatched_results.append(unmatched_results)

        return ClusterResults(
            dataframe=concat(
                [matched_results.dataframe]
                + [cluster_result.dataframe for cluster_result in all_unmatched_results]
            ).convert_dtypes(dtype_backend="pyarrow"),
            run_name=results.run_name,
            description=results.description,
            left=results.left,
            right=results.right,
        )
    else:
        return matched_results
