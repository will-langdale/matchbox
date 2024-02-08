import os

import pytest
from pandas import DataFrame
from sqlalchemy.orm import Session

from cmf import make_deduper, to_clusters
from cmf.data import Models
from cmf.dedupers import NaiveDeduper

data_test_params = [
    (
        f"{os.getenv('SCHEMA')}.crn",
        "query_clean_crn",
        [f"{os.getenv('SCHEMA')}_crn_company_name", f"{os.getenv('SCHEMA')}_crn_crn"],
        # 1000 unique items repeated three times
        1000,
        3000,
        # Unordered pairs of sets of three, so (3 choose 2) = 3, * 1000 = 3000
        3000,
        # TO UPDATE
        0000,
    ),
    (
        f"{os.getenv('SCHEMA')}.duns",
        "query_clean_duns",
        [
            f"{os.getenv('SCHEMA')}_duns_company_name",
            f"{os.getenv('SCHEMA')}_duns_duns",
        ],
        # 500 unique items with no duplication
        500,
        500,
        # No duplicates
        0,
        # TO UPDATE
        0000,
    ),
    (
        f"{os.getenv('SCHEMA')}.cdms",
        "query_clean_cdms",
        [f"{os.getenv('SCHEMA')}_cdms_crn", f"{os.getenv('SCHEMA')}_cdms_cdms"],
        # 1000 unique items repeated two times
        1000,
        2000,
        # Unordered pairs of sets of two, so (2 choose 2) = 1, * 1000 = 1000
        1000,
        # TO UPDATE
        0000,
    ),
]


def make_naive_dd_settings(
    source, data_fixture, fields, unique_n, curr_n, tgt_prob_n, tgt_clus_n
):
    return {"id": "data_sha1", "unique_fields": fields}


deduper_test_params = [("naive", NaiveDeduper, make_naive_dd_settings)]


@pytest.mark.parametrize(
    "source, data_fixture, fields, unique_n, curr_n, tgt_prob_n, tgt_clus_n",
    data_test_params,
)
@pytest.mark.parametrize(
    "deduper_name, deduper_class, build_deduper_settings", deduper_test_params
)
def test_dedupers(
    # Fixtures
    db_engine,
    db_clear_models,
    # Data params
    source,
    data_fixture,
    fields,
    unique_n,
    curr_n,
    tgt_prob_n,
    tgt_clus_n,
    # Methodology params
    deduper_name,
    deduper_class,
    build_deduper_settings,
    # Pytest
    request,
):
    """Runs all deduper methodologies over exemplar tables.

    Tests:
        * That the input data is as expected
        * That the data is deduplicated correctly
        * That the deduplicated probabilities are inserted correctly
        * The the correct number of clusters are resolved
        * That the resolved clusters are inserted correctly
    """
    df = request.getfixturevalue(data_fixture)

    # 1. Input data is as expected

    assert isinstance(df, DataFrame)
    assert df.shape[0] == curr_n

    # 2. Data is deduplicated correctly

    deduper_settings = build_deduper_settings(
        source, data_fixture, fields, unique_n, curr_n, tgt_prob_n, tgt_clus_n
    )

    deduper = make_deduper(
        dedupe_run_name=f"{deduper_name}_{source}",
        description=f"Testing dedupe of {source} with {deduper_name} method",
        deduper=deduper_class,
        deduper_settings=deduper_settings,
        data_source=source,
        data=df,
    )

    deduped = deduper()

    deduped_df = deduped.to_df()
    deduped_df_with_source = deduped.inspect_with_source(
        left_data=df, left_key="data_sha1", right_data=df, right_key="data_sha1"
    )

    assert isinstance(deduped_df, DataFrame)
    assert deduped_df.shape[0] == tgt_prob_n
    for field in fields:
        assert deduped_df_with_source[field + "_x"].equals(
            deduped_df_with_source[field + "_y"]
        )

    # 3. Deduplicated probabilities are inserted correctly

    deduped.to_cmf(engine=db_engine[1])

    with Session(db_engine[1]) as session:
        model = session.query(Models).filter_by(name=f"{deduper_name}_{source}").first()
        proposed_dedupes = model.proposes_dedupes

    assert len(proposed_dedupes) == tgt_prob_n

    # 4. Correct number of clusters are resolved

    clusters_merge_only = to_clusters(results=deduped, key="data_sha1")

    clusters_merge_only_df = clusters_merge_only.to_df()

    assert clusters_merge_only_df

    clusters_all = to_clusters(df, results=deduped, key="data_sha1")

    clusters_all_df = clusters_all.to_df()

    assert clusters_all_df

    # 5. Resolved clusters are inserted correctly

    # i. Clean up

    db_clear_models(db_engine)
