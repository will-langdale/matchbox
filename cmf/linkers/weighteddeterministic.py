from typing import List

import duckdb
from pandas import DataFrame
from pydantic import BaseModel, Field, field_validator

from cmf.helpers import comparison
from cmf.linkers.make_linker import Linker, LinkerSettings


class WeightedComparison(BaseModel):
    """A valid comparison and a weight to give it."""

    comparison: str = Field(
        description="""
            A valid ON clause to compare fields between the left and 
            the right data.

            Use left.field and right.field to refer to columns in the 
            respective sources.

            For example:

            "left.name = right.name and left.company_id = right.id"
        """
    )
    weight: int = Field(
        description="""
            A weight to give this comparison. Use 1 for all comparisons to give
            uniform weight to each.
        """
    )

    @field_validator("comparison")
    @classmethod
    def validate_comparison(cls, v: str) -> str:
        comp_val = comparison(v)
        return comp_val


class WeightedDeterministicSettings(LinkerSettings):
    """
    A data class to enforce the Weighted linker's settings dictionary shape.
    """

    weighted_comparisons: List[WeightedComparison] = Field(
        description="""
            A list of tuples in the form of a comparison, and a weight.

            Example:

                >>> {
                ...     left_id: "cluster_sha1",
                ...     right_id: "cluster_sha1",
                ...     weighted_comparisons: [
                ...         ("l.company_name = r.company_name", 1),
                ...         ("l.postcode = r.postcode", 1),
                ...         ("l.company_id = r.company_id", 2)
                ...     ],
                ...     threshold: 0.8
                ... }
        """
    )
    threshold: float = Field(
        description="""
            The probability above which matches will be kept. 
            
            Inclusive, so a value of 1 will keep only exact matches across all 
            comparisons.
        """,
        ge=0,
        le=1,
    )


class WeightedDeterministicLinker(Linker):
    settings: WeightedDeterministicSettings

    @classmethod
    def from_settings(
        cls, left_id: str, right_id: str, weighted_comparisons: List, threshold: float
    ) -> "WeightedDeterministicLinker":
        settings = WeightedDeterministicSettings(
            left_id=left_id,
            right_id=right_id,
            weighted_comparisons=[
                WeightedComparison(comparison=comparison[0], weight=comparison[1])
                for comparison in weighted_comparisons
            ],
            threshold=threshold,
        )
        return cls(settings=settings)

    def prepare(self, left: DataFrame, right: DataFrame) -> None:
        pass

    def link(self, left: DataFrame, right: DataFrame) -> DataFrame:
        left_df = left.copy()  # NoQA: F841. It's used below but ruff can't detect
        right_df = right.copy()  # NoQA: F841. It's used below but ruff can't detect

        match_subquery = []
        weights = []

        for weighted_comparison in self.settings.weighted_comparisons:
            match_subquery.append(
                f"""
                    select distinct on (list_sort([raw.left_id, raw.right_id]))
                        raw.left_id,
                        raw.right_id,
                        1 * {weighted_comparison.weight} as probability
                    from (
                        select
                            l.{self.settings.left_id} as left_id,
                            r.{self.settings.right_id} as right_id,
                        from
                            left_df l
                        inner join right_df r on
                            {weighted_comparison.comparison}
                    ) raw
                """
            )
            weights.append(weighted_comparison.weight)

        match_subquery = " union all ".join(match_subquery)
        total_weight = sum(weights)

        return (
            duckdb.sql(
                f"""
                    select
                        matches.left_id,
                        matches.right_id,
                        sum(matches.probability) / {total_weight} as probability
                    from
                        ({match_subquery}) matches
                    group by
                        matches.left_id,
                        matches.right_id
                    having
                        sum(matches.probability) / 
                            {total_weight} >= {self.settings.threshold};
                """
            )
            .arrow()
            .to_pandas()
        )  # correctly returns bytes -- .df() returns bytesarray
