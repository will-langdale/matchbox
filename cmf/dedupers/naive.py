from typing import List

import duckdb
from pandas import DataFrame
from pydantic import Field

from cmf.dedupers.make_deduper import Deduper, DeduperSettings


class NaiveSettings(DeduperSettings):
    """
    A data class to enforce the Naive deduper's settings dictionary shape
    """

    unique_fields: List[str] = Field(
        description="""\
            A list of columns that will form a unique, deduplicated record\
        """
    )


class NaiveDeduper(Deduper):
    settings: NaiveSettings

    @classmethod
    def from_settings(cls, id: str, unique_fields: List[str]) -> "NaiveDeduper":
        settings = NaiveSettings(id=id, unique_fields=unique_fields)
        return cls(settings=settings)

    def prepare(self, data: DataFrame) -> None:
        pass

    def dedupe(self, data: DataFrame) -> DataFrame:
        df = data.copy()

        join_clause = []
        for field in self.settings.unique_fields:
            join_clause.append(f"l.{field} = r.{field}")
        join_clause_compiled = " and ".join(join_clause)

        # Generate a new PK to remove row self-match but ALLOW true duplicate
        # rows where all data items are identical in the source
        df["_unique_e4003b"] = range(df.shape[0])

        return duckdb.sql(
            f"""
            select distinct on (list_sort([raw.left_id, raw.right_id]))
                raw.left_id,
                raw.right_id,
                1 as probability
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
            ) raw;
        """
        ).df()
