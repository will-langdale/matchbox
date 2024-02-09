import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import rustworkx as rx
from pandas import DataFrame, concat
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import Engine, Table, bindparam, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session

from cmf.data import utils as du
from cmf.data.clusters import Clusters
from cmf.data.data import SourceData
from cmf.data.db import ENGINE
from cmf.data.dedupe import DDupeContains, Dedupes
from cmf.data.link import LinkContains, Links
from cmf.data.models import Models, ModelsFrom

logic_logger = logging.getLogger("cmf_logic")


class CMFSourceDataError(Exception):
    """Data doesn't exist in the SourceData table."""


class CMFSourceDatasetError(Exception):
    """Tables not found in wider database, outside of the framework."""


class CMFClusterError(Exception):
    """Data doesn't exist in the Cluster table."""


class CMFModelError(Exception):
    """Data doesn't exist in the Model table."""

    def __init__(self, message, model):
        super().__init__(message)


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

    def _validate_tables(self, engine: Engine = ENGINE) -> bool:
        """Validates existence of left and right tables in wider database."""
        try:
            db_left_schema, db_left_table = du.get_schema_table_names(
                full_name=self.left, validate=True
            )
            _ = du.string_to_table(
                db_schema=db_left_schema, db_table=db_left_table, engine=engine
            )
            db_right_schema, db_right_table = du.get_schema_table_names(
                full_name=self.right, validate=True
            )
            _ = du.string_to_table(
                db_schema=db_right_schema, db_table=db_right_table, engine=engine
            )
        except NoSuchTableError:
            return False

        return True

    def _validate_sources(self, engine: Engine = ENGINE) -> bool:
        """Validates existence of left and right models in CMF database."""
        stmt_left = select(Models.name).where(Models.name == self.left)
        stmt_right = select(Models.name).where(Models.name == self.right)

        with Session(engine) as session:
            res_left = session.execute(stmt_left).scalar()
            res_right = session.execute(stmt_right).scalar()

        if res_left is not None and res_right is not None:
            return True
        else:
            return False

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
        """Writes the model to the CMF."""
        with Session(engine) as session:
            if deduplicates is None:
                # Linker
                # Construct model SHA1 from parent model SHA1s
                left_sha1 = du.model_name_to_sha1(self.left, engine=engine)
                right_sha1 = du.model_name_to_sha1(self.right, engine=engine)
                model_sha1 = du.list_to_value_ordered_sha1(
                    [self.run_name, left_sha1, right_sha1]
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
        metadata = f"{self._get_results_type()}, {self.run_name}"

        if self.left == self.right:
            # Deduper
            # Write model
            logic_logger.info(f"[{metadata}] Registering model")
            self._model_to_cmf(
                deduplicates=du.table_name_to_sha1(self.left, engine=engine),
                engine=engine,
            )

            # Write data
            if self.dataframe.shape[0] == 0:
                logic_logger.info(f"[{metadata}] No deduplication data to insert")
            else:
                logic_logger.info(f"[{metadata}] Writing deduplication data")
                self._deduper_to_cmf(engine=engine)
        else:
            # Linker
            # Write model
            logic_logger.info(f"[{metadata}] Registering model")
            self._model_to_cmf(engine=engine)

            # Write data
            if self.dataframe.shape[0] == 0:
                logic_logger.info(f"[{metadata}] No link data to insert")
            else:
                logic_logger.info(f"[{metadata}] Writing link data")
                self._linker_to_cmf(engine=engine)

        logic_logger.info(f"[{metadata}] Complete!")


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
            .filter(["left_id", "right_id"])
            .map(str)
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
        )[["model", "left", "left_id", "right", "right_id", "probability"]]

        return df

    def _prep_to_cmf(self, df: DataFrame, engine: Engine = ENGINE) -> Dict[str, Any]:
        """Transforms data to dictionary and calculates SHA-1 hash."""
        pre_prep_df = df.copy()
        cols = ["left_id", "right_id"]

        # Verify data is in the CMF
        pre_prep_df[cols] = pre_prep_df[cols].map(bytes)

        for col in cols:
            data_unique = pre_prep_df[col].unique().tolist()

            with Session(engine) as session:
                data_inner_join = (
                    session.query(SourceData)
                    .filter(
                        SourceData.sha1.in_(
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
                    raise CMFSourceDataError(
                        f"Some items in {col} don't exist in SourceData table. "
                        "Did you use data_sha1 as your ID when deduplicating?"
                    )

        # Transform for insert
        pre_prep_df["sha1"] = du.columns_to_value_ordered_sha1(
            data=self.dataframe, columns=cols
        )
        pre_prep_df = pre_prep_df.rename(
            columns={"left_id": "left", "right_id": "right"}
        )

        return pre_prep_df.to_dict("records")

    def _deduper_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a deduper to the CMF database.

        * Turns data into a left, right, sha1, probability dataframe
        * Upserts data then queries it back as SQLAlchemy objects
        * Attaches these objects to the model

        Raises:
            CMFSourceDatasetError is source tables aren't in the wider database
            CMFModelError if current model wasn't inserted correctly
        """
        probabilities_to_add = self._prep_to_cmf(self.dataframe, engine=engine)

        if not self._validate_tables(engine=engine):
            raise CMFSourceDatasetError(
                "Source tables not found in wider database. Check your "
                "deduplication sources."
            )

        with Session(engine) as session:
            # Add probabilities
            # Get model, including adding association proxy classes to session
            model = session.query(Models).filter_by(name=self.run_name).first()
            if model is None:
                raise CMFModelError(
                    "Results model not found in database", self.run_name
                )

            # Insert any new Dedupe nodes, without probabilities
            ins_dd_stmt = insert(Dedupes)
            ins_dd_stmt = ins_dd_stmt.on_conflict_do_nothing(
                index_elements=[Dedupes.sha1]
            )
            _ = session.execute(
                ins_dd_stmt,
                [
                    {k: v for k, v in dd.items() if k != "probability"}
                    for dd in probabilities_to_add
                ],
            )
            session.commit()

            # Add probabilities to the appropriate model.proposes_dedupes
            to_insert = (
                session.query(Dedupes)
                .filter(
                    Dedupes.sha1.in_(
                        bindparam(
                            "ins_sha1s",
                            [dd["sha1"] for dd in probabilities_to_add],
                            expanding=True,
                        )
                    )
                )
                .all()
            )

            model.proposes_dedupes.clear()

            for dd, r in zip(to_insert, probabilities_to_add):
                model.proposes_dedupes[dd] = r["probability"]

            session.commit()

    def _linker_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a linker to the CMF database.

        * Turns data into a left, right, sha1, probability dataframe
        * Upserts data then queries it back as SQLAlchemy objects
        * Attaches these objects to the model

        Raises:
            CMFModelError
                * If source models weren't found in the database
                * If current model wasn't inserted correctly
        """
        probabilities_to_add = self._prep_to_cmf(self.dataframe, engine=engine)

        if not self._validate_sources(engine=engine):
            raise CMFModelError(
                "Source models not found in database. Check your link sources.",
                [self.left, self.right],
            )

        with Session(engine) as session:
            # Add probabilities
            # Get model, including adding association proxy classes to session
            model = session.query(Models).filter_by(name=self.run_name).first()
            if model is None:
                raise CMFModelError(
                    "Results model not found in database", self.run_name
                )

            # Insert any new Link nodes, without probabilities
            ins_li_stmt = insert(Links)
            ins_li_stmt = ins_li_stmt.on_conflict_do_nothing(
                index_elements=[Links.sha1]
            )
            _ = session.execute(
                ins_li_stmt,
                [
                    {k: v for k, v in li.items() if k != "probability"}
                    for li in probabilities_to_add
                ],
            )
            session.commit()

            # Add probabilities to the appropriate model.proposes_links
            to_insert = (
                session.query(Links)
                .filter(
                    Links.sha1.in_(
                        bindparam(
                            "ins_sha1s",
                            [li["sha1"] for li in probabilities_to_add],
                            expanding=True,
                        )
                    )
                )
                .all()
            )

            model.proposes_links.clear()

            for dd, r in zip(to_insert, probabilities_to_add):
                model.proposes_links[dd] = r["probability"]

            session.commit()


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
        return self.dataframe.copy()

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
            CMFModelError if model wasn't inserted correctly
        """
        Contains = contains_class
        with Session(engine) as session:
            # Get model, including adding association proxy classes to session
            model = session.query(Models).filter_by(name=self.run_name).first()
            if model is None:
                raise CMFModelError(
                    "Results model not found in database", self.run_name
                )

            # Add new cluster nodes
            clusters_to_add = [
                {"sha1": edge} for edge in self.dataframe.parent.drop_duplicates()
            ]

            ins_stmt = insert(Clusters)
            ins_stmt = ins_stmt.on_conflict_do_nothing(index_elements=[Clusters.sha1])

            session.execute(ins_stmt, clusters_to_add)
            session.commit()

            to_insert = (
                session.query(Clusters)
                .filter(
                    Clusters.sha1.in_(
                        bindparam(
                            "ins_sha1s",
                            [cl["sha1"] for cl in clusters_to_add],
                            expanding=True,
                        )
                    )
                )
                .all()
            )

            # Add model endorsement of clusters
            model.creates.clear()
            model.creates = to_insert

            session.commit()

            # Add new contains edges
            ins_stmt = insert(Contains)
            ins_stmt = ins_stmt.on_conflict_do_update(
                index_elements=[Contains.parent, Contains.child],
                set_=ins_stmt.excluded,
            )

            session.execute(ins_stmt, self.dataframe.to_dict("records"))
            session.commit()

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
        clusters (ClusterResults): a ClusterResults generated by a linker
        data (DataFrame): cleaned data that went into the model
        key (str): the column that was matched, usually data_sha1 or cluster_sha1
        threshold (float): the value above which to consider probabilities true

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
        dataframe=DataFrame(no_parent),
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
        results.to_df()
        .query("probability > @threshold")
        .filter(["left_id", "right_id"])
        .map(bytes)
        .stack()
    )

    G = rx.PyGraph()
    added = {}

    for edge in all_edges.groupby(level=0):
        edge_idx = []
        for i, sha1 in edge[1].items():
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
        dataframe=DataFrame(res),
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
            ),
            run_name=results.run_name,
            description=results.description,
            left=results.left,
            right=results.right,
        )
    else:
        return matched_results
