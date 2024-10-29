import logging
from abc import ABC, abstractmethod
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
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import Table

if TYPE_CHECKING:
    from matchbox.models.models import Model
else:
    Model = Any

logic_logger = logging.getLogger("mb_logic")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


class ResultsBaseDataclass(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: DataFrame
    model: Model

    _expected_fields: list[str]

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


class ProbabilityResults(ResultsBaseDataclass):
    """Probabilistic matches produced by linkers and dedupers.

    Inherits the following attributes from ResultsBaseDataclass.

    _expected_fields enforces the shape of the dataframe.

    Args:
        dataframe (DataFrame): the DataFrame holding the results
        model (Model): x
    """

    _expected_fields: list[str] = [
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
            left=self.model.metadata.left_source,
            right=self.model.metadata.right_source,
            model=self.model.metadata.model_name,
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

        # Preprocess the dataframe
        pre_prep_df = self.dataframe[["left_id", "right_id", "probability"]].copy()
        pre_prep_df[["left_id", "right_id"]] = pre_prep_df[
            ["left_id", "right_id"]
        ].astype("binary[pyarrow]")
        pre_prep_df["sha1"] = columns_to_value_ordered_hash(
            data=pre_prep_df, columns=["left_id", "right_id"]
        )
        pre_prep_df["sha1"] = pre_prep_df["sha1"].astype("binary[pyarrow]")

        return {
            Probability(hash=row[0], left=row[1], right=row[2], probability=row[3])
            for row in pre_prep_df[
                ["sha1", "left_id", "right_id", "probability"]
            ].to_numpy()
        }


class ClusterResults(ResultsBaseDataclass):
    """Cluster data produced by using to_clusters on ProbabilityResults.

    Inherits the following attributes from ResultsBaseDataclass.

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    probabilities: ProbabilityResults
    clusters: ClusterResults

    @inject_backend
    def to_matchbox(self, backend: MatchboxDBAdapter) -> None:
        """Writes the results to the Matchbox database."""
        self.model.insert_model()
        self.model.clusters = self


def process_components(
    G: rx.PyGraph,
    current_pairs: set[bytes],
    added: dict[bytes, int],
    pair_children: dict[bytes, list[bytes]],
) -> list[tuple[bytes, list[bytes], int]]:
    """
    Process connected components in the current graph state.

    Identifies which 2-item components have merged into larger components.

    Returns:
        List of (parent_hash, children, size) for components > 2 items
        where children includes both individual items and parent hashes
    """
    new_components = []
    component_with_size = [
        (component, len(component)) for component in rx.connected_components(G)
    ]

    for component, size in component_with_size:
        if size <= 2:
            continue

        # Get all node hashes in component
        node_hashes = [G.get_node_data(node) for node in component]

        # Find which 2-item parents are part of this component
        component_pairs = {
            pair
            for pair in current_pairs
            if any(G.has_node(added[h]) for h in node_hashes)
        }

        # Children are individual nodes not in pairs, plus the pair parents
        children = component_pairs | {
            h
            for h in node_hashes
            if not any(h in pair_children[p] for p in component_pairs)
        }

        parent_hash = list_to_value_ordered_hash(sorted(children))
        new_components.append((parent_hash, list(children)))

    return new_components


def to_clusters(results: ProbabilityResults) -> ClusterResults:
    """
    Takes a models probabilistic outputs and turns them into clusters.

    Performs connected components at decreasing thresholds from 1.0 to return every
    possible component in a hierarchical tree.

    * Stores all two-item components with their original probabilities
    * For larger components, stores the individual items and two-item parent hashes
        as children, with a new parent hash

    Args:
        results: ProbabilityResults object

    Returns:
        ClusterResults object
    """
    G = rx.PyGraph()
    added: dict[bytes, int] = {}
    pair_children: dict[bytes, list[bytes]] = {}
    current_pairs: set[bytes] = set()
    seen_larger: set[bytes] = set()

    clusters = {"parent": [], "child": [], "threshold": []}

    # 1. Create all 2-item components with original probabilities
    initial_edges = (
        results.dataframe.filter(["left_id", "right_id", "probability"])
        .astype({"left_id": "binary[pyarrow]", "right_id": "binary[pyarrow]"})
        .itertuples(index=False, name=None)
    )

    for left, right, prob in initial_edges:
        for hash_val in (left, right):
            if hash_val not in added:
                idx = G.add_node(hash_val)
                added[hash_val] = idx

        children = sorted([left, right])
        parent_hash = list_to_value_ordered_hash(children)

        pair_children[parent_hash] = children
        current_pairs.add(parent_hash)

        clusters["parent"].extend([parent_hash] * 2)
        clusters["child"].extend(children)
        clusters["threshold"].extend([prob] * 2)

        G.add_edge(added[left], added[right], None)

    # 2. Process at each probability threshold
    sorted_probabilities = sorted(
        results.dataframe["probability"].unique(), reverse=True
    )

    for threshold in sorted_probabilities:
        # Find new larger components at this threshold
        new_components = process_components(G, current_pairs)

        # Add new components to results
        for parent_hash, children in new_components:
            if parent_hash not in seen_larger:
                seen_larger.add(parent_hash)
                clusters["parent"].extend([parent_hash] * len(children))
                clusters["child"].extend(children)
                clusters["threshold"].extend([threshold] * len(children))

    return ClusterResults(
        dataframe=DataFrame(clusters).convert_dtypes(dtype_backend="pyarrow"),
        model=results.model,
    )
