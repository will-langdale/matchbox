from typing import List

import duckdb
from pandas import DataFrame
from pydantic import BaseModel, Field

from cmf.dedupers.make_deduper import Deduper


class NaiveSettings(BaseModel):
    """
    A data class to enforce the Naive deduper's settings dictionary shape
    """

    id: str = Field(description="A unique ID column in the table to dedupe")
    unique_fields: List[str] = Field(
        description="""\
            A list of columns that will form a unique, deduplicated record\
        """
    )


class Naive(Deduper):
    settings: NaiveSettings

    @classmethod
    def from_settings(cls, id: str, unique_fields: List[str]) -> "Naive":
        settings = NaiveSettings(id=id, unique_fields=unique_fields)
        return cls(settings=settings)

    def dedupe(self, data: DataFrame) -> DataFrame:
        join_clause = []
        for field in self.settings.unique_fields:
            join_clause.append(f"l.{field} = r.{field}")
        join_clause_compiled = " and ".join(join_clause)

        return duckdb.sql(
            f"""
            select
                l.{self.settings.id}::text as target_id,
                r.{self.settings.id}::text as source_id,
                1 as probability
            from
                exp_sample_cleaned l
            inner join exp_sample_cleaned r on
                (
                    {join_clause_compiled}
                ) and
                    l.{self.settings.id} != r.{self.settings.id};
        """
        ).df()
