"""A linking methodology based on a deterministic set of conditions."""

from typing import Iterable, Type

import polars as pl
from pandas import DataFrame
from pydantic import Field, field_validator

from matchbox.client.helpers import comparison
from matchbox.client.models.linkers.base import Linker, LinkerSettings


class DeterministicSettings(LinkerSettings):
    """A data class to enforce the Deterministic linker's settings dictionary shape."""

    comparisons: Iterable[str] = Field(
        description="""
            An iterable of valid ON clause to compare fields between the left and 
            the right data.

            Use left.field and right.field to refer to columns in the respective 
            sources.

            Each comparison will be treated as OR logic, but more efficiently than using
            an OR condition in the SQL WHERE clause.

            For example:

            [   
                "left.company_number = right.company_number",
                "left.name = right.name and left.postcode = right.postcode",
            ]
        """,
    )

    @field_validator("comparisons")
    @classmethod
    def validate_comparison(cls, v: Iterable[str]) -> Iterable[str]:
        """Validate the comparison string."""
        return [comparison(comp_val) for comp_val in v]


class DeterministicLinker(Linker):
    """A deterministic linker that links based on a set of boolean conditions."""

    settings: DeterministicSettings

    _id_dtype_l: Type = None
    _id_dtype_r: Type = None

    @classmethod
    def from_settings(
        cls, left_id: str, right_id: str, comparisons: str
    ) -> "DeterministicLinker":
        """Create a DeterministicLinker from a settings dictionary."""
        settings = DeterministicSettings(
            left_id=left_id, right_id=right_id, comparisons=comparisons
        )
        return cls(settings=settings)

    def prepare(self, left: DataFrame, right: DataFrame) -> None:
        """Prepare the linker for linking."""
        pass

    def link(self, left: DataFrame, right: DataFrame) -> DataFrame:
        """Link the left and right dataframes."""
        left_pl = pl.from_pandas(left).lazy()  # noqa: F841
        right_pl = pl.from_pandas(right).lazy()  # noqa: F841

        subqueries = []
        for i, condition in enumerate(self.settings.comparisons):
            subquery = f"""
                SELECT
                    l_{i}.{self.settings.left_id} AS left_id,
                    r_{i}.{self.settings.right_id} AS right_id,
                    1.0 AS probability
                FROM left_pl l_{i}
                INNER JOIN right_pl r_{i}
                    ON {condition}
            """
            subqueries.append(subquery)

        union_query = " UNION ALL ".join(subqueries)

        final_query = f"""
            SELECT DISTINCT 
                final.left_id, 
                final.right_id, 
                final.probability
            FROM ({union_query}) as final
        """

        return pl.sql(final_query).collect().to_pandas()
