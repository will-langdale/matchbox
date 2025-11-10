"""A linking methodology based on a deterministic set of conditions."""

import polars as pl
from pydantic import Field, field_validator

from matchbox.client.models import comparison
from matchbox.client.models.linkers.base import Linker, LinkerSettings


class DeterministicSettings(LinkerSettings):
    """A data class to enforce the Deterministic linker's settings dictionary shape."""

    comparisons: list[str] | list[list[str]] = Field(
        description="""
            Comparison rules for matching.
            
            Can be specified as:
            - A flat list of strings: All comparisons applied in parallel (OR logic)
            - A nested list of lists: Sequential rounds of matching
            
            Flat list (parallel):
            [
                "left.company_number = right.company_number",
                "left.name = right.name",
            ]
            All comparisons applied to full datasets, results unioned.
            
            Nested list (sequential rounds):
            [
                [
                    "left.company_number = right.company_number",
                    "left.name = right.name",
                ],
                [
                    "left.name_normalised = right.name_normalised",
                    "left.website = right.website",
                ],
            ]
            Each inner list is a "round". Within each round, comparisons use OR 
            logic. After each round, matched records are removed from the pool 
            before the next round.
            
            Use left.field and right.field to refer to columns in the respective 
            sources.
        """,
    )

    @field_validator("comparisons", mode="before")
    @classmethod
    def validate_comparison(
        cls, value: str | list[str] | list[list[str]]
    ) -> list[list[str]]:
        """Normalise to list of lists format."""
        # Single string -> [[string]]
        if isinstance(value, str):
            return [[comparison(value)]]

        # Empty list
        if not value:
            raise ValueError("comparisons cannot be empty")

        # Check if flat list of strings
        if all(isinstance(v, str) for v in value):
            # Flat list -> wrap in outer list for single round (parallel mode)
            return [[comparison(v) for v in value]]

        # List of lists
        if all(isinstance(v, list) for v in value):
            # Validate all inner elements are strings
            for round_idx, round_comparisons in enumerate(value):
                if not round_comparisons:
                    raise ValueError(f"Round {round_idx} cannot be empty")
                if not all(isinstance(c, str) for c in round_comparisons):
                    raise ValueError(
                        f"Round {round_idx} must contain only strings, "
                        f"got: {[type(c).__name__ for c in round_comparisons]}"
                    )
            return [[comparison(c) for c in round_comps] for round_comps in value]

        raise ValueError(
            "comparisons must be a string, list of strings, or list of lists of strings"
        )


class DeterministicLinker(Linker):
    """A deterministic linker that links based on a set of boolean conditions.

    Supports both parallel matching (single round) and sequential matching
    (multiple rounds where matched records are removed after each round).
    """

    settings: DeterministicSettings

    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        """Prepare the linker for linking."""
        pass

    def link(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        """Link the left and right dataframes.

        If comparisons is a flat list, applies all comparisons in parallel.
        If comparisons is a nested list, applies each round sequentially,
        removing matched records from the pool after each round.
        """
        all_matches: list[pl.DataFrame] = []
        remaining_left: pl.DataFrame = left
        remaining_right: pl.DataFrame = right

        # Iterate through rounds
        for round_idx, round_comparisons in enumerate(self.settings.comparisons):
            if remaining_left.is_empty() or remaining_right.is_empty():
                break

            # Apply all comparisons in this round (parallel/OR logic)
            round_matches: pl.DataFrame = self._link_round(
                remaining_left, remaining_right, round_comparisons, round_idx
            )

            if not round_matches.is_empty():
                all_matches.append(round_matches)

                # Remove matched records from pools for next round
                matched_left_ids: pl.DataFrame = round_matches.select(
                    "left_id"
                ).unique()
                matched_right_ids: pl.DataFrame = round_matches.select(
                    "right_id"
                ).unique()

                remaining_left = remaining_left.join(
                    matched_left_ids,
                    left_on=self.settings.left_id,
                    right_on="left_id",
                    how="anti",
                )
                remaining_right = remaining_right.join(
                    matched_right_ids,
                    left_on=self.settings.right_id,
                    right_on="right_id",
                    how="anti",
                )

        return self._finalise_results(all_matches)

    def _link_round(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        comparisons: list[str],
        round_idx: int,
    ) -> pl.DataFrame:
        """Apply all comparisons in a round using OR logic.

        All comparisons within a round are applied to the same datasets
        and results are unioned together.
        """
        left_pl: pl.LazyFrame = left.lazy()  # noqa: F841
        right_pl: pl.LazyFrame = right.lazy()  # noqa: F841

        subqueries: list[str] = []
        for comp_idx, condition in enumerate(comparisons):
            subquery: str = f"""
                SELECT
                    l_{round_idx}_{comp_idx}.{self.settings.left_id} AS left_id,
                    r_{round_idx}_{comp_idx}.{self.settings.right_id} AS right_id,
                    1.0 AS probability
                FROM left_pl l_{round_idx}_{comp_idx}
                INNER JOIN right_pl r_{round_idx}_{comp_idx}
                    ON {condition}
            """
            subqueries.append(subquery)

        union_query: str = " UNION ALL ".join(subqueries)

        final_query: str = f"""
            SELECT DISTINCT 
                final.left_id, 
                final.right_id, 
                final.probability
            FROM ({union_query}) as final
        """

        return pl.sql(final_query).collect()

    def _finalise_results(self, all_matches: list[pl.DataFrame]) -> pl.DataFrame:
        """Combine matches from all rounds and ensure correct schema."""
        if all_matches:
            return pl.concat(all_matches).with_columns(
                pl.col("probability").cast(pl.Float32)
            )
        else:
            # Return empty dataframe with correct schema
            return pl.DataFrame(
                {"left_id": [], "right_id": [], "probability": []}
            ).with_columns(pl.col("probability").cast(pl.Float32))
