import duckdb
from pandas import DataFrame
from pydantic import Field, field_validator

from cmf.helpers import comparison
from cmf.linkers.make_linker import Linker, LinkerSettings


class DeterministicSettings(LinkerSettings):
    """
    A data class to enforce the Naive deduper's settings dictionary shape
    """

    comparisons: str = Field(
        description=""""\
            A valid ON clause to compare fields between the left and \
            the right data.

            Use left.field and right.field to refer to columns in the \
            respective sources.

            For example:

            "left.name = right.name and left.company_id = right.id"
        """
    )

    @field_validator("comparisons")
    @classmethod
    def validate_comparison(cls, v: str) -> str:
        comp_val = comparison(v)
        return comp_val


class DeterministicLinker(Linker):
    settings: DeterministicSettings

    @classmethod
    def from_settings(
        cls, left_id: str, right_id: str, comparisons: str
    ) -> "DeterministicLinker":
        settings = DeterministicSettings(
            left_id=left_id, right_id=right_id, comparisons=comparisons
        )
        return cls(settings=settings)

    def prepare(self, left: DataFrame, right: DataFrame) -> None:
        pass

    def link(self, left: DataFrame, right: DataFrame) -> DataFrame:
        left_df = left  # NoQA: F841. It's used below but ruff can't detect
        right_df = right  # NoQA: F841. It's used below but ruff can't detect
        return duckdb.sql(
            f"""
            select distinct on (list_sort([raw.left_id, raw.right_id]))
                raw.left_id,
                raw.right_id,
                1 as probability
            from (
                select
                    l.{self.settings.left_id} as left_id,
                    r.{self.settings.right_id} as right_id,
                from
                    left_df l
                inner join right_df r on
                    {self.settings.comparisons}
            ) raw;
        """
        ).df()
