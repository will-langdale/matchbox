import os
from typing import Any, Callable, Dict, List, Type

from pydantic import BaseModel, Field

from cmf.dedupers import NaiveDeduper
from cmf.dedupers.make_deduper import Deduper


class DedupeTestParams(BaseModel):
    """Data class for raw dataset testing parameters and attributes."""

    source: str = Field(description="SQL reference for the source table")
    fixture: str = Field(description="pytest fixture of the clean data")
    fields: List[str] = Field(description="Data fields to select and work with")
    unique_n: int = Field(description="Unique items in this data")
    curr_n: int = Field(description="Current row count of this data")
    tgt_prob_n: int = Field(description="Expected count of generated probabilities")
    tgt_clus_n: int = Field(description="Expected count of resolved clusters")


Model = Type[Deduper]
DataSettings = Callable[[DedupeTestParams], Dict[str, Any]]


class ModelTestParams(BaseModel):
    """Data class for model testing parameters and attributes."""

    name: str = Field(description="Model name")
    cls: Model = Field(description="Model class")
    build_settings: DataSettings = Field(
        description=(
            "A function that takes an object of type DedupeTestParams or "
            "LinkTestParams and returns an appropriate settings dictionary "
            "for this deduper."
        )
    )


dedupe_test_params = [
    DedupeTestParams(
        source=f"{os.getenv('SCHEMA')}.crn",
        fixture="query_clean_crn",
        fields=[
            f"{os.getenv('SCHEMA')}_crn_company_name",
            f"{os.getenv('SCHEMA')}_crn_crn",
        ],
        # 1000 unique items repeated three times
        unique_n=1000,
        curr_n=3000,
        # Unordered pairs of sets of three, so (3 choose 2) = 3, * 1000 = 3000
        tgt_prob_n=3000,
        tgt_clus_n=1000,
    ),
    DedupeTestParams(
        source=f"{os.getenv('SCHEMA')}.duns",
        fixture="query_clean_duns",
        fields=[
            f"{os.getenv('SCHEMA')}_duns_company_name",
            f"{os.getenv('SCHEMA')}_duns_duns",
        ],
        # 500 unique items with no duplication
        unique_n=500,
        curr_n=500,
        # No duplicates
        tgt_prob_n=0,
        tgt_clus_n=0,
    ),
    DedupeTestParams(
        source=f"{os.getenv('SCHEMA')}.cdms",
        fixture="query_clean_cdms",
        fields=[
            f"{os.getenv('SCHEMA')}_cdms_crn",
            f"{os.getenv('SCHEMA')}_cdms_cdms",
        ],
        # 1000 unique items repeated two times
        unique_n=1000,
        curr_n=2000,
        # Unordered pairs of sets of two, so (2 choose 2) = 1, * 1000 = 1000
        tgt_prob_n=1000,
        tgt_clus_n=1000,
    ),
]


def make_naive_dd_settings(data: DedupeTestParams) -> Dict[str, Any]:
    return {"id": "data_sha1", "unique_fields": data.fields}


deduper_test_params = [
    ModelTestParams(
        name="naive", cls=NaiveDeduper, build_settings=make_naive_dd_settings
    )
]
