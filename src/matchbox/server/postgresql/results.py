import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import rustworkx as rx
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame, concat
from pg_bulk_ingest import Delete, Upsert, ingest
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import (
    Engine,
    Table,
    delete,
)
from sqlalchemy.orm import Session

from matchbox.server.base import Cluster, MatchboxDBAdapter, Probability
from matchbox.server.exceptions import MatchboxDBDataError
from matchbox.server.postgresql import utils as du
from matchbox.server.postgresql.clusters import Clusters, clusters_association
from matchbox.server.postgresql.db import ENGINE
from matchbox.server.postgresql.dedupe import DDupeContains
from matchbox.server.postgresql.link import LinkContains
from matchbox.server.postgresql.models import Models

logic_logger = logging.getLogger("mb_logic")

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

    @model_validator(mode="after")
    def _check_dataframe(self) -> Table:
        """Verifies the table contains the expected fields."""
        table_fields = sorted(self.dataframe.columns)
        expected_fields = sorted(self._expected_fields)

        if table_fields != expected_fields:
            raise ValueError(f"Expected {expected_fields}. \n" f"Found {table_fields}.")

        return self

    @abstractmethod
    def inspect_with_source(self) -> DataFrame:
        """Enriches the results with the source data."""
        return

    @abstractmethod
    def to_df(self) -> DataFrame:
        """Returns the results as a DataFrame."""
        return

    @abstractmethod
    def to_records(self) -> list[Probability | Cluster]:
        """Returns the results as a list of records suitable for insertion."""
        return

    def to_cmf(self, backend: MatchboxDBAdapter) -> None:
        """Writes the results to the CMF database."""
        if self.left == self.right:
            # Deduper
            backend.insert_model(
                model=self.run_name,
                left=self.left,
                description=self.description,
            )

            model = backend.get_model(model=self.run_name)

            model.insert_probabilities(
                probabilites=self.to_records(),
                probability_type="deduplications",
                batch_size=backend.settings.batch_size,
            )
        else:
            # Linker
            backend.insert_model(
                model=self.run_name,
                left=self.left,
                right=self.right,
                description=self.description,
            )

            model = backend.get_model(model=self.run_name)

            model.insert_probabilities(
                probabilites=self.to_records(),
                probability_type="links",
                batch_size=backend.settings.batch_size,
            )


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

    def to_records(self, backend: MatchboxDBAdapter | None) -> list[Probability]:
        """Returns the results as a list of records suitable for insertion.

        If given a backend, will validate the records against the database.
        """
        # Optional validation
        if backend:
            if self.left == self.right:
                hash_type = "data"  # Deduper
            else:
                hash_type = "cluster"  # Linker

            backend.validate_hashes(
                hashes=self.dataframe.left_id.unique().tolist(),
                hash_type=hash_type,
            )
            backend.validate_hashes(
                hashes=self.dataframe.right_id.unique().tolist(),
                hash_type=hash_type,
            )

        # Prep and return
        pre_prep_df = self.dataframe.copy()
        cols = ["left_id", "right_id"]
        pre_prep_df[cols] = pre_prep_df[cols].astype("binary[pyarrow]")
        pre_prep_df["sha1"] = du.columns_to_value_ordered_sha1(
            data=self.dataframe, columns=cols
        )
        pre_prep_df.sha1 = pre_prep_df.sha1.astype("binary[pyarrow]")
        pre_prep_df = pre_prep_df.rename(
            columns={"left_id": "left", "right_id": "right"}
        )

        pre_prep_df = pre_prep_df[["sha1", "left", "right", "probability"]]

        return [
            Probability(
                sha1=sha1,
                left=left,
                right=right,
                probability=probability,
            )
            for sha1, left, right, probability in self.dataframe.itertuples(
                index=False, name=None
            )
        ]


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

    def _to_mb_logic(
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
            MatchboxDBDataError if model wasn't inserted correctly
        """
        Contains = contains_class
        with Session(engine) as session:
            # Add clusters
            # Get model
            model = session.query(Models).filter_by(name=self.run_name).first()
            model_sha1 = model.sha1

            if model is None:
                raise MatchboxDBDataError(source=Models, data=self.run_name)

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
        self._to_mb_logic(contains_class=DDupeContains, engine=engine)

    def _linker_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a linker to the CMF database."""
        self._to_mb_logic(contains_class=LinkContains, engine=engine)


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
