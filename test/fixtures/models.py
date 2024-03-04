import os
from typing import Any, Callable, Dict, Type, Union

import splink.duckdb.comparison_library as cl
from pydantic import BaseModel, Field
from splink.duckdb import blocking_rule_library as brl
from splink.duckdb.linker import DuckDBLinker

from cmf.dedupers import NaiveDeduper
from cmf.dedupers.make_deduper import Deduper
from cmf.linkers import DeterministicLinker, SplinkLinker, WeightedDeterministicLinker
from cmf.linkers.make_linker import Linker


class DedupeTestParams(BaseModel):
    """Data class for raw dataset testing parameters and attributes."""

    source: str = Field(description="SQL reference for the source table")
    fixture: str = Field(description="pytest fixture of the clean data")
    fields: Dict[str, str] = Field(
        description=(
            "Data fields to select and work with. The key is the name of the "
            "field as it comes out of the database, the value is what it should be "
            "renamed to for models that require field names be identical, like the "
            "SplinkLinker."
        )
    )
    unique_n: int = Field(description="Unique items in this data")
    curr_n: int = Field(description="Current row count of this data")
    tgt_prob_n: int = Field(description="Expected count of generated probabilities")
    tgt_clus_n: int = Field(description="Expected count of resolved clusters")


class LinkTestParams(BaseModel):
    """Data class for deduped dataset testing parameters and attributes."""

    source_l: str = Field(description="SQL reference for the left source table")
    fixture_l: str = Field(description="pytest fixture of the clean left data")
    fields_l: Dict[str, str] = Field(
        description=(
            "Left data fields to select and work with. The key is the name of the "
            "field as it comes out of the database, the value is what it should be "
            "renamed to for models that require field names be identical, like the "
            "SplinkLinker."
        )
    )
    curr_n_l: int = Field(description="Current row count of the left data")

    source_r: str = Field(description="SQL reference for the right source table")
    fixture_r: str = Field(description="pytest fixture of the clean right data")
    fields_r: Dict[str, str] = Field(
        description=(
            "Right data fields to select and work with. The key is the name of the "
            "field as it comes out of the database, the value is what it should be "
            "renamed to for models that require field names be identical, like the "
            "SplinkLinker."
        )
    )
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
    rename_fields: bool = Field(
        description=(
            "Whether fields should be coerced to have matching names, as required "
            "by some Linkers and Dedupers."
        )
    )


dedupe_data_test_params = [
    DedupeTestParams(
        source=f"{os.getenv('SCHEMA')}.crn",
        fixture="query_clean_crn",
        fields={
            f"{os.getenv('SCHEMA')}_crn_company_name": "company_name",
            f"{os.getenv('SCHEMA')}_crn_crn": "crn",
        },
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
        fields={
            f"{os.getenv('SCHEMA')}_duns_company_name": "company_name",
            f"{os.getenv('SCHEMA')}_duns_duns": "duns",
        },
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
        fields={
            f"{os.getenv('SCHEMA')}_cdms_crn": "crn",
            f"{os.getenv('SCHEMA')}_cdms_cdms": "cdms",
        },
        # 1000 unique items repeated two times
        unique_n=1000,
        curr_n=2000,
        # Unordered pairs of sets of two, so (2 choose 2) = 1, * 1000 = 1000
        tgt_prob_n=1000,
        tgt_clus_n=1000,
    ),
]


link_data_test_params = [
    LinkTestParams(
        # Left
        source_l=f"naive_{os.getenv('SCHEMA')}.crn",
        fixture_l="query_clean_crn_deduped",
        fields_l={f"{os.getenv('SCHEMA')}_crn_company_name": "company_name"},
        curr_n_l=3000,
        # Right
        source_r=f"naive_{os.getenv('SCHEMA')}.duns",
        fixture_r="query_clean_duns_deduped",
        fields_r={f"{os.getenv('SCHEMA')}_duns_company_name": "company_name"},
        curr_n_r=500,
        # Check
        unique_n=1000,
        # Remember these are deduped: 1000 unique in the left, 500 in the right
        tgt_prob_n=500,
        tgt_clus_n=500,
    ),
    LinkTestParams(
        # Left
        source_l=f"naive_{os.getenv('SCHEMA')}.cdms",
        fixture_l="query_clean_cdms_deduped",
        fields_l={
            f"{os.getenv('SCHEMA')}_cdms_crn": "crn",
        },
        curr_n_l=2000,
        # Right
        source_r=f"naive_{os.getenv('SCHEMA')}.crn",
        fixture_r="query_clean_crn_deduped",
        fields_r={f"{os.getenv('SCHEMA')}_crn_crn": "crn"},
        curr_n_r=3000,
        # Check
        unique_n=1000,
        # Remember these are deduped: 1000 unique in the left, 1000 in the right
        tgt_prob_n=1000,
        tgt_clus_n=1000,
    ),
]


def make_naive_dd_settings(data: DedupeTestParams) -> Dict[str, Any]:
    return {"id": "data_sha1", "unique_fields": list(data.fields.keys())}


dedupe_model_test_params = [
    ModelTestParams(
        name="naive",
        cls=NaiveDeduper,
        build_settings=make_naive_dd_settings,
        rename_fields=False,
    )
]


def make_deterministic_li_settings(data: LinkTestParams) -> Dict[str, Any]:
    comparisons = []

    for field_l, field_r in zip(data.fields_l.keys(), data.fields_r.keys()):
        comparisons.append(f"l.{field_l} = r.{field_r}")

    return {
        "left_id": "cluster_sha1",
        "right_id": "cluster_sha1",
        "comparisons": " and ".join(comparisons),
    }


def make_splink_li_settings(data: LinkTestParams) -> Dict[str, Any]:
    fields_l = data.fields_l.values()
    fields_r = data.fields_r.values()
    if set(fields_l) != set(fields_r):
        raise ValueError("Splink requires fields have identical names")
    else:
        fields = list(fields_l)

    comparisons = []

    for field in fields:
        comparisons.append(f"l.{field} = r.{field}")

    linker_training_functions = [
        {
            "function": "estimate_probability_two_random_records_match",
            "arguments": {
                "deterministic_matching_rules": comparisons,
                "recall": 1,
            },
        },
        {
            "function": "estimate_u_using_random_sampling",
            "arguments": {"max_pairs": 1e4},
        },
    ]

    # The m parameter is 1 because we're testing in a deterministic system, and
    # many of these tests only have one field, so we can't use expectation
    # maximisation to estimate. For testing raw functionality, fine to use 1
    linker_settings = {
        "retain_matching_columns": False,
        "retain_intermediate_calculation_columns": False,
        "blocking_rules_to_generate_predictions": [
            brl.block_on(field) for field in fields
        ],
        "comparisons": [
            cl.exact_match(field, m_probability_exact_match=1) for field in fields
        ],
    }

    return {
        "left_id": "cluster_sha1",
        "right_id": "cluster_sha1",
        "linker_class": DuckDBLinker,
        "linker_training_functions": linker_training_functions,
        "linker_settings": linker_settings,
        "threshold": None,
    }


def make_weighted_deterministic_li_settings(data: LinkTestParams) -> Dict[str, Any]:
    weighted_comparisons = []

    for field_l, field_r in zip(data.fields_l, data.fields_r):
        weighted_comparisons.append((f"l.{field_l} = r.{field_r}", 1))

    return {
        "left_id": "cluster_sha1",
        "right_id": "cluster_sha1",
        "weighted_comparisons": weighted_comparisons,
        "threshold": 1,
    }


link_model_test_params = [
    ModelTestParams(
        name="deterministic",
        cls=DeterministicLinker,
        build_settings=make_deterministic_li_settings,
        rename_fields=False,
    ),
    ModelTestParams(
        name="weighted_deterministic",
        cls=WeightedDeterministicLinker,
        build_settings=make_weighted_deterministic_li_settings,
        rename_fields=False,
    ),
    ModelTestParams(
        name="splink",
        cls=SplinkLinker,
        build_settings=make_splink_li_settings,
        rename_fields=True,
    ),
]
