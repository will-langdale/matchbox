import logging

from dotenv import find_dotenv, load_dotenv
from matchbox.common.hash import (
    columns_to_value_ordered_hash,
)
from matchbox.models.models import Model
from matchbox.server.base import MatchboxDBAdapter, inject_backend
from matchbox.server.models import Probability
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import Table

logic_logger = logging.getLogger("mb_logic")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


class ProbabilityResults(BaseModel):
    """Probabilistic matches produced by linkers and dedupers.

    Inherits the following attributes from ResultsBaseDataclass.

    _expected_fields enforces the shape of the dataframe.

    Args:
        dataframe (DataFrame): the DataFrame holding the results
        model_name (str): the name of the model generating the results
        description (str): a description of the model generating the results
        left (str): the source dataset or source model for the left side of
            the comparison
        right (str): the source dataset or source model for the right side of
            the comparison
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: DataFrame
    model: Model

    _expected_fields: list[str] = [
        "left_id",
        "right_id",
        "probability",
    ]

    @model_validator(mode="after")
    def _check_dataframe(self) -> Table:
        """Verifies the table contains the expected fields."""
        table_fields = sorted(self.dataframe.columns)
        expected_fields = sorted(self._expected_fields)

        if table_fields != expected_fields:
            raise ValueError(f"Expected {expected_fields}. \n" f"Found {table_fields}.")

        return self

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
    def to_records(self, backend: MatchboxDBAdapter | None) -> list[Probability]:
        """Returns the results as a list of records suitable for insertion.

        If given a backend, will validate the records against the database.
        """
        # Optional validation
        if backend:
            backend.validate_hashes(
                hashes=self.dataframe.left_id.unique().tolist(),
            )
            backend.validate_hashes(
                hashes=self.dataframe.right_id.unique().tolist(),
            )

        # Preprocess the dataframe
        pre_prep_df = self.dataframe[["left_id", "right_id", "probability"]].copy()
        pre_prep_df[["left_id", "right_id"]] = pre_prep_df[
            ["left_id", "right_id"]
        ].astype("binary[pyarrow]")
        pre_prep_df["sha1"] = columns_to_value_ordered_hash(
            data=pre_prep_df, columns=["left_id", "right_id"]
        )
        pre_prep_df["sha1"] = pre_prep_df["sha1"].astype("binary[pyarrow]")

        return [
            Probability(hash=row[0], left=row[1], right=row[2], probability=row[3])
            for row in pre_prep_df[
                ["sha1", "left_id", "right_id", "probability"]
            ].to_numpy()
        ]

    @inject_backend
    def to_matchbox(self, backend: MatchboxDBAdapter) -> None:
        """Writes the results to the Matchbox database."""
        self.model.insert_model()
        self.model.insert_probabilities(
            probabilities=self.to_records(backend=backend),
            batch_size=backend.settings.batch_size,
        )
