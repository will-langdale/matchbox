import os
from typing import Any, Callable, Dict, List, Type, Union

from pydantic import BaseModel, Field

from cmf.dedupers import NaiveDeduper
from cmf.dedupers.make_deduper import Deduper
from cmf.linkers import DeterministicLinker
from cmf.linkers.make_linker import Linker


class DedupeTestParams(BaseModel):
    """Data class for raw dataset testing parameters and attributes."""

    source: str = Field(description="SQL reference for the source table")
    fixture: str = Field(description="pytest fixture of the clean data")
    fields: List[str] = Field(description="Data fields to select and work with")
    unique_n: int = Field(description="Unique items in this data")
    curr_n: int = Field(description="Current row count of this data")
    tgt_prob_n: int = Field(description="Expected count of generated probabilities")
    tgt_clus_n: int = Field(description="Expected count of resolved clusters")


class LinkTestParams(BaseModel):
    """Data class for deduped dataset testing parameters and attributes."""

    source_l: str = Field(description="SQL reference for the left source table")
    fixture_l: str = Field(description="pytest fixture of the clean left data")
    fields_l: List[str] = Field(description="Left data fields to select and work with")
    curr_n_l: int = Field(description="Current row count of the left data")

    source_r: str = Field(description="SQL reference for the right source table")
    fixture_r: str = Field(description="pytest fixture of the clean right data")
    fields_r: List[str] = Field(description="Right data fields to select and work with")
    curr_n_r: int = Field(description="Current row count of the right data")

    unique_n: int = Field(description="Unique items in the merged data")
    tgt_prob_n: int = Field(description="Expected count of generated probabilities")
    tgt_clus_n: int = Field(description="Expected count of resolved clusters")


Model = Type[Union[Deduper, Linker]]
DataSettings = Callable[[Union[DedupeTestParams, LinkTestParams]], Dict[str, Any]]


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


merge_test_params = [
    LinkTestParams(
        # Left
        source_l=f"{os.getenv('SCHEMA')}.crn",
        fixture_l="query_clean_crn_deduped",
        fields_l=[
            f"{os.getenv('SCHEMA')}_crn_company_name",
            f"{os.getenv('SCHEMA')}_crn_crn",
        ],
        curr_n_l=3000,
        # Right
        source_r=f"{os.getenv('SCHEMA')}.duns",
        fixture_r="query_clean_duns_deduped",
        fields_r=[
            f"{os.getenv('SCHEMA')}_duns_company_name",
            f"{os.getenv('SCHEMA')}_duns_duns",
        ],
        curr_n_r=500,
        # Check
        unique_n=1000,
        tgt_prob_n=3000,
        tgt_clus_n=1000,
    ),
    LinkTestParams(
        # Left
        source_l=f"{os.getenv('SCHEMA')}.cdms",
        fixture_l="query_clean_cdms_deduped",
        fields_l=[
            f"{os.getenv('SCHEMA')}_cdms_crn",
            f"{os.getenv('SCHEMA')}_cdms_cdms",
        ],
        curr_n_l=2000,
        # Right
        source_r=f"{os.getenv('SCHEMA')}.crn",
        fixture_r="query_clean_crn_deduped",
        fields_r=[
            f"{os.getenv('SCHEMA')}_crn_company_name",
            f"{os.getenv('SCHEMA')}_crn_crn",
        ],
        curr_n_r=3000,
        # Check
        unique_n=1000,
        tgt_prob_n=6000,
        tgt_clus_n=1000,
    ),
    LinkTestParams(
        # Left
        source_l=f"{os.getenv('SCHEMA')}.duns",
        fixture_l="query_clean_duns_deduped",
        fields_l=[
            f"{os.getenv('SCHEMA')}_duns_company_name",
            f"{os.getenv('SCHEMA')}_duns_duns",
        ],
        curr_n_l=500,
        # Right
        source_r=f"{os.getenv('SCHEMA')}.cdms",
        fixture_r="query_clean_cdms_deduped",
        fields_r=[
            f"{os.getenv('SCHEMA')}_cdms_crn",
            f"{os.getenv('SCHEMA')}_cdms_cdms",
        ],
        curr_n_r=2000,
        # Check
        unique_n=1000,
        tgt_prob_n=2000,
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


def make_deterministic_li_settings(data: LinkTestParams) -> Dict[str, Any]:
    comparisons = []

    for field_l, field_r in zip(data.fields_l, data.fields_r):
        comparisons.append(f"l.{field_l} = r.{field_r}")

    return {
        "left_id": "cluster_sha1",
        "right_id": "cluster_sha1",
        "comparisons": " and ".join(comparisons),
    }


linker_test_params = [
    ModelTestParams(
        name="deterministic",
        cls=DeterministicLinker,
        build_settings=make_deterministic_li_settings,
    )
]
