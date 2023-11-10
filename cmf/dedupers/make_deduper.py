from cmf.data.probabilities import ProbabilityResults

from pydantic import BaseModel
from pandas import DataFrame
from typing import Dict, List, Any, Callable
from abc import ABC


class Deduper(BaseModel, ABC):
    settings: Dict[str, Any]

    @classmethod
    def from_settings(cls) -> "Deduper":
        raise NotImplementedError(
            """\
            Must implement method to instantiate from settings \
            -- consider creating a pydantic model to enforce shape.
        """
        )

    def dedupe(self, data) -> DataFrame:
        raise NotImplementedError("Must implement dedupe method")


def make_deduper(
    dedupe_run_name: str,
    description: str,
    deduper: Deduper,
    data: DataFrame,
    data_source: str,
    dedupe_settings=Dict[str, List],
) -> Callable[[DataFrame], ProbabilityResults]:
    deduper_instance = deduper.from_settings(**dedupe_settings)

    def dedupe(data: DataFrame = data) -> ProbabilityResults:
        return ProbabilityResults(
            dataframe=deduper_instance.dedupe(data=data),
            run_name=dedupe_run_name,
            description=description,
            target=data_source,
            source=data_source,
        )

    return dedupe
