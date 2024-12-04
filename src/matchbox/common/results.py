import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any, List

import rustworkx as rx
from dotenv import find_dotenv, load_dotenv
from matchbox.common.hash import (
    columns_to_value_ordered_hash,
    list_to_value_ordered_hash,
)
from matchbox.server.base import MatchboxDBAdapter, inject_backend
from matchbox.server.models import Cluster, Probability
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from sqlalchemy import Table

if TYPE_CHECKING:
    from matchbox.models.models import Model, ModelMetadata
else:
    Model = Any

logic_logger = logging.getLogger("mb_logic")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


class ModelType(StrEnum):
    """Enumeration of supported model types."""

    LINKER = "linker"
    DEDUPER = "deduper"


class ModelMetadata(BaseModel):
    """Metadata for a model."""

    name: str
    description: str
    type: ModelType
    left_source: str
    right_source: str | None = None  # Only used for linker models


class ResultsBaseDataclass(BaseModel, ABC):
    """Base class for results dataclasses.

    Model is required during construction and calculation, but not when loading
    from storage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: DataFrame
    model: Model | None = None
    metadata: ModelMetadata

    _expected_fields: list[str]

    @model_validator(mode="after")
    def _check_dataframe(self) -> Table:
        """Verifies the table contains the expected fields."""
        table_fields = set(self.dataframe.columns)
        expected_fields = set(self._expected_fields)

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


class ProbabilityResults(ResultsBaseDataclass):
    """Probabilistic matches produced by linkers and dedupers.

    There are pairs of records/clusters with a probability of being a match.
    The hash is the hash of the sorted left and right ids.

    _expected_fields enforces the shape of the dataframe.

    Args:
        dataframe (DataFrame): the DataFrame holding the results
        model (Model): x
    """

    _expected_fields: list[str] = [
        "hash",
        "left_id",
        "right_id",
        "probability",
    ]

    @field_validator("dataframe")
    @classmethod
    def add_hash(cls, dataframe: DataFrame) -> DataFrame:
        """Adds a hash column to the dataframe if it doesn't already exist."""
        if "hash" not in dataframe.columns:
            dataframe[["left_id", "right_id"]] = dataframe[
                ["left_id", "right_id"]
            ].astype("binary[pyarrow]")
            dataframe["hash"] = columns_to_value_ordered_hash(
                data=dataframe, columns=["left_id", "right_id"]
            )
            dataframe["hash"] = dataframe["hash"].astype("binary[pyarrow]")
        return dataframe[["hash", "left_id", "right_id", "probability"]]

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
            left=self.model.metadata.left_source,
            right=self.model.metadata.right_source,
            model=self.metadata.name,
        ).convert_dtypes(dtype_backend="pyarrow")[
            ["model", "left", "left_id", "right", "right_id", "probability"]
        ]

        return df

    @inject_backend
    def to_records(self, backend: MatchboxDBAdapter | None) -> set[Probability]:
        """Returns the results as a list of records suitable for insertion.

        If given a backend, will validate the records against the database.
        """
        # Optional validation
        if backend:
            backend.validate_hashes(hashes=self.dataframe.left_id.unique().tolist())
            backend.validate_hashes(hashes=self.dataframe.right_id.unique().tolist())

        return {
            Probability(hash=row[0], left=row[1], right=row[2], probability=row[3])
            for row in self.dataframe[
                ["hash", "left_id", "right_id", "probability"]
            ].to_numpy()
        }


class ClusterResults(ResultsBaseDataclass):
    """Cluster data produced by using to_clusters on ProbabilityResults.

    This is the connected components of the probabilistic matches at every
    threshold of probabilitity. The parent is the hash of the sorted children.

    _expected_fields enforces the shape of the dataframe.

    Args:
        dataframe (DataFrame): the DataFrame holding the results
        model (Model): x
    """

    _expected_fields: List[str] = ["parent", "child", "threshold"]

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
            .filter(["parent", "child", "probability"])
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

    def to_records(self) -> set[Cluster]:
        """Returns the results as a list of records suitable for insertion."""
        # Preprocess the dataframe
        pre_prep_df = (
            self.dataframe[["parent", "child", "threshold"]]
            .groupby(["parent", "threshold"], as_index=False)["child"]
            .agg(list)
            .copy()
        )

        return {
            Cluster(parent=row[0], children=row[1], threshold=row[2])
            for row in pre_prep_df.to_numpy()
        }


class Results(BaseModel):
    """A container for the results of a model run.

    Contains all the information any backend will need to store the results.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    probabilities: ProbabilityResults
    clusters: ClusterResults

    @inject_backend
    def to_matchbox(self, backend: MatchboxDBAdapter) -> None:
        """Writes the results to the Matchbox database."""
        if self.probabilities.model != self.clusters.model:
            raise ValueError("Probabilities and clusters must be from the same model.")

        self.clusters.model.insert_model()
        self.clusters.model.results = self


def to_clusters(results: ProbabilityResults) -> ClusterResults:
    """
    Converts probabilities into a list of connected components formed at each threshold.

    Returns:
        ClusterResults sorted by threshold descending.
    """
    G = rx.PyGraph()
    added: dict[bytes, int] = {}
    components: dict[str, list] = {"parent": [], "child": [], "threshold": []}

    # Sort probabilities descending and group by probability
    edges_df = (
        results.dataframe.sort_values("probability", ascending=False)
        .filter(["left_id", "right_id", "probability"])
        .astype(
            {"left_id": "large_binary[pyarrow]", "right_id": "large_binary[pyarrow]"}
        )
    )

    # Get unique probability thresholds, sorted
    thresholds = edges_df["probability"].unique()

    # Process edges grouped by probability threshold
    for prob in thresholds:
        threshold_edges = edges_df[edges_df["probability"] == prob]
        # Get state before adding this batch of edges
        old_components = {frozenset(comp) for comp in rx.connected_components(G)}

        # Add all nodes and edges at this probability threshold
        edge_values = threshold_edges[["left_id", "right_id"]].values
        for left, right in edge_values:
            for hash_val in (left, right):
                if hash_val not in added:
                    idx = G.add_node(hash_val)
                    added[hash_val] = idx

            G.add_edge(added[left], added[right], None)

        new_components = {frozenset(comp) for comp in rx.connected_components(G)}
        changed_components = new_components - old_components

        # For each changed component, add ALL members at current threshold
        for comp in changed_components:
            children = sorted([G.get_node_data(n) for n in comp])
            parent = list_to_value_ordered_hash(children)

            components["parent"].extend([parent] * len(children))
            components["child"].extend(children)
            components["threshold"].extend([prob] * len(children))

    return ClusterResults(
        dataframe=DataFrame(components).convert_dtypes(dtype_backend="pyarrow"),
        model=results.model,
        metadata=results.metadata,
    )
