from cmf.dedupers.make_deduper import Deduper
from pandas import DataFrame
import duckdb
from pydantic import BaseModel, Field
from typing import List


class NaiveSettings(BaseModel):
    """
    A data class to enforce Naive's settings dictionary shape
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
        unique_fields = ", ".join(self.settings.unique_fields)
        # TODO: This needs to return PROBABILITIES
        return duckdb.sql(
            f"""
            select distinct on ({unique_fields})
                {self.settings.id},
                {unique_fields}
            from
                data
            order by
                {unique_fields};
        """
        ).df()
