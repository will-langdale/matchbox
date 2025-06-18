"""A deduplication methodology based on a deterministic set of conditions."""

from typing import Type

import duckdb
import polars as pl
from pydantic import Field

from matchbox.client.models.dedupers.base import Deduper, DeduperSettings


class NaiveSettings(DeduperSettings):
    """A data class to enforce the Naive deduper's settings dictionary shape."""

    unique_fields: list[str] = Field(
        description="A list of fields that will form a unique, deduplicated record"
    )


class NaiveDeduper(Deduper):
    """A simple deduper that deduplicates based on a set of boolean conditions."""

    settings: NaiveSettings

    _id_dtype: Type = None

    @classmethod
    def from_settings(cls, id: str, unique_fields: list[str]) -> "NaiveDeduper":
        """Create a NaiveDeduper from a settings dictionary."""
        settings = NaiveSettings(id=id, unique_fields=unique_fields)
        return cls(settings=settings)

    def prepare(self, data: pl.DataFrame) -> None:
        """Prepare the deduper for deduplication."""
        pass

    def dedupe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Deduplicate the dataframe."""
        self._id_dtype = type(data[self.settings.id][0])

        df = data.clone()

        join_clause = []
        for field in self.settings.unique_fields:
            join_clause.append(f"l.{field} = r.{field}")
        join_clause_compiled = " and ".join(join_clause)

        # Generate a key to remove row self-match but ALLOW true duplicate
        # rows where all data items are identical in the source
        df = df.with_row_index("_unique_e4003b")

        # We also need to suppress raw.left_id = raw.right_id in cases where
        # we're deduplicating an unnested array of primary keys
        sql = f"""
            select distinct on (list_sort([raw.left_id, raw.right_id]))
                raw.left_id,
                raw.right_id,
                1.0 as probability
            from (
                select
                    l.{self.settings.id} as left_id,
                    r.{self.settings.id} as right_id
                from
                    df l
                inner join df r on
                    (
                        {join_clause_compiled}
                    ) and
                        l._unique_e4003b != r._unique_e4003b
            ) raw
                where raw.left_id != raw.right_id;
        """
        df_arrow = duckdb.sql(sql).arrow()
        result_pl = pl.from_arrow(df_arrow)
        del df_arrow

        # Ensure ID columns are uint64 (required by Results validation)
        if result_pl.schema["left_id"] != pl.UInt64:
            result_pl = result_pl.with_columns(
                [
                    pl.col("left_id").cast(pl.UInt64),
                    pl.col("right_id").cast(pl.UInt64),
                ]
            )

        return result_pl
