import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Hashable, TypeVar

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from sqlalchemy import Table

from matchbox.common.db import Cluster, Probability
from matchbox.common.hash import columns_to_value_ordered_hash
from matchbox.server.base import MatchboxDBAdapter, inject_backend

if TYPE_CHECKING:
    from matchbox.client.models.models import Model, ModelMetadata
else:
    Model = Any

T = TypeVar("T", bound=Hashable)

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
        "id",
        "left_id",
        "right_id",
        "probability",
    ]

    @field_validator("dataframe", mode="before")
    @classmethod
    def results_to_hash(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Adds an ID column to the dataframe if it doesn't already exist.

        * Reattaches hashes from the backend
        * Uses them to create the new ID column
        """
        id_exists = "id" in dataframe.columns
        l_is_int = pd.api.types.is_integer_dtype(dataframe["left_id"])
        r_is_int = pd.api.types.is_integer_dtype(dataframe["right_id"])

        if id_exists and l_is_int and r_is_int:
            return dataframe

        @inject_backend
        def _make_id_hasher(backend: MatchboxDBAdapter):
            """Closure for converting int columns to hash using a lookup."""
            lookup: dict[int, bytes] = {}

            def _hash_column(df: pd.DataFrame, column_name: str) -> None:
                hashed_column = f"{column_name}_hashed"
                unique_ids = df[column_name].unique().tolist()

                lookup.update(backend.cluster_id_to_hash(ids=unique_ids))

                df[hashed_column] = (
                    df[column_name].map(lookup).astype("binary[pyarrow]")
                )
                df.drop(columns=[column_name], inplace=True)
                df.rename(columns={hashed_column: column_name}, inplace=True)

            return _hash_column

        hash_column = _make_id_hasher()

        # Update lookup with left_id, then convert to hash
        if l_is_int:
            hash_column(df=dataframe, column_name="left_id")

        # Update lookup with right_id, then convert to hash
        if r_is_int:
            hash_column(df=dataframe, column_name="right_id")

        # Create ID column if it doesn't exist and hash the values
        if not id_exists:
            dataframe[["left_id", "right_id"]] = dataframe[
                ["left_id", "right_id"]
            ].astype("binary[pyarrow]")
            dataframe["id"] = columns_to_value_ordered_hash(
                data=dataframe, columns=["left_id", "right_id"]
            )
            dataframe["id"] = dataframe["id"].astype("binary[pyarrow]")

        return dataframe

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
                ["id", "left_id", "right_id", "probability"]
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

    _expected_fields: list[str] = ["parent", "child", "threshold"]

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
