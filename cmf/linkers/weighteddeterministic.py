from typing import List, Type

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

            "left.company_name = right.company_name"
        """
    )
    weight: float = Field(
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

    Example:

        >>> {
        ...     left_id: "cluster_sha1",
        ...     right_id: "cluster_sha1",
        ...     weighted_comparisons: [
        ...         ("l.company_name = r.company_name", .7),
        ...         ("l.postcode = r.postcode", .7),
        ...         ("l.company_id = r.company_id", 1)
        ...     ],
        ...     threshold: 0.8
        ... }
    """

    weighted_comparisons: List[WeightedComparison] = Field(
        description="A list of tuples in the form of a comparison, and a weight."
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

    _id_dtype_l: Type = None
    _id_dtype_r: Type = None

    @classmethod
    def from_settings(
        cls, left_id: str, right_id: str, weighted_comparisons: List, threshold: float
    ) -> "WeightedDeterministicLinker":
        settings = WeightedDeterministicSettings(
            left_id=left_id,
            right_id=right_id,
            # No change in weighted_comparisons data, just validates the input list
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
        self._id_dtype_l = type(left[self.settings.left_id][0])
        self._id_dtype_r = type(right[self.settings.right_id][0])

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

        res = duckdb.sql(
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
        ).df()

        # Convert bytearray back to bytes
        return res.assign(
            left_id=lambda df: df.left_id.apply(self._id_dtype_l),
            right_id=lambda df: df.right_id.apply(self._id_dtype_r),
        )
