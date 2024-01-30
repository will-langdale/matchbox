from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from pandas import DataFrame
from pydantic import BaseModel

from cmf.data.results import ProbabilityResults


class Deduper(BaseModel, ABC):
    settings: Dict[str, Any]

    @classmethod
    @abstractmethod
    def from_settings(cls) -> "Deduper":
        raise NotImplementedError(
            """\
            Must implement method to instantiate from settings \
            -- consider creating a pydantic model to enforce shape.
        """
        )

    @abstractmethod
    def prepare(self, data: DataFrame) -> None:
        return

    @abstractmethod
    def dedupe(self, data: DataFrame) -> DataFrame:
        return


def make_deduper(
    dedupe_run_name: str,
    description: str,
    deduper: Deduper,
    deduper_settings: Dict[str, Any],
    data: DataFrame,
    data_source: str,
) -> Callable[[DataFrame], ProbabilityResults]:
    deduper_instance = deduper.from_settings(**deduper_settings)
    deduper_instance.prepare(data)

    def dedupe(data: DataFrame = data) -> ProbabilityResults:
        return ProbabilityResults(
            dataframe=deduper_instance.dedupe(data=data),
            run_name=dedupe_run_name,
            description=description,
            left=data_source,
            right=data_source,
            validate_as="tables",
        )

    return dedupe
