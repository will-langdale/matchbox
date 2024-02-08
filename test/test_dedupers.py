import os

import pytest
from pandas import DataFrame, Series, concat
from sqlalchemy.orm import Session

from cmf import make_deduper, process, query
from cmf.clean import company_name
from cmf.data import Models
from cmf.data import utils as du
from cmf.dedupers import NaiveDeduper
from cmf.helpers import cleaner, cleaners, selector


@pytest.fixture(scope="function")
def query_clean_crn(db_engine):
    # Select
    select_crn = selector(
        table=f"{os.getenv('SCHEMA')}.crn",
        fields=["crn", "company_name"],
        engine=db_engine[1],
    )

    crn = query(
        selector=select_crn, model=None, return_type="pandas", engine=db_engine[1]
    )

    # Clean
    col_prefix = f"{os.getenv('SCHEMA')}_crn_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_crn = cleaners(cleaner_name)

    crn_cleaned = process(data=crn, pipeline=cleaner_crn)

    return crn_cleaned


@pytest.fixture(scope="function")
def query_clean_duns(db_engine):
    # Select
    select_duns = selector(
        table=f"{os.getenv('SCHEMA')}.duns",
        fields=["duns", "company_name"],
        engine=db_engine[1],
    )

    duns = query(
        selector=select_duns, model=None, return_type="pandas", engine=db_engine[1]
    )

    # Clean
    col_prefix = f"{os.getenv('SCHEMA')}_duns_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_duns = cleaners(cleaner_name)

    duns_cleaned = process(data=duns, pipeline=cleaner_duns)

    return duns_cleaned


@pytest.fixture(scope="function")
def query_clean_cdms(db_engine):
    # Select
    select_cdms = selector(
        table=f"{os.getenv('SCHEMA')}.cdms",
        fields=["crn", "cdms"],
        engine=db_engine[1],
    )

    cdms = query(
        selector=select_cdms, model=None, return_type="pandas", engine=db_engine[1]
    )

    # No cleaning needed, see original data
    return cdms


def test_sha1_conversion(all_companies):
    """Tests SHA1 conversion works as expected."""
    sha1_series_1 = du.columns_to_value_ordered_sha1(
        data=all_companies,
        columns=["id", "company_name", "address", "crn", "duns", "cdms"],
    )

    assert isinstance(sha1_series_1, Series)
    assert len(sha1_series_1) == all_companies.shape[0]

    all_companies_reordered_top = (
        all_companies.head(500)
        .rename(
            columns={
                "company_name": "address",
                "address": "company_name",
                "duns": "crn",
                "crn": "duns",
            }
        )
        .filter(["id", "company_name", "address", "crn", "duns", "cdms"])
    )

    all_companies_reodered = concat(
        [all_companies_reordered_top, all_companies.tail(500)]
    )

    sha1_series_2 = du.columns_to_value_ordered_sha1(
        data=all_companies_reodered,
        columns=["id", "company_name", "address", "crn", "duns", "cdms"],
    )

    assert sha1_series_1.equals(sha1_series_2)


data_test_params = [
    (
        f"{os.getenv('SCHEMA')}.crn",
        "query_clean_crn",
        [f"{os.getenv('SCHEMA')}_crn_company_name", f"{os.getenv('SCHEMA')}_crn_crn"],
        3000,
        # 1000 unique items repeated three times
        # Unordered pairs of sets of three, so (3 choose 2) = 3, * 1000 = 3000
        3000,
    ),
    (
        f"{os.getenv('SCHEMA')}.duns",
        "query_clean_duns",
        [
            f"{os.getenv('SCHEMA')}_duns_company_name",
            f"{os.getenv('SCHEMA')}_duns_duns",
        ],
        500,
        # every row is unique: no duplicates
        0,
    ),
    (
        f"{os.getenv('SCHEMA')}.cdms",
        "query_clean_cdms",
        [f"{os.getenv('SCHEMA')}_cdms_crn", f"{os.getenv('SCHEMA')}_cdms_cdms"],
        2000,
        # 1000 unique items repeated two times
        # Unordered pairs of sets of two, so (2 choose 2) = 1, * 1000 = 1000
        1000,
    ),
]


def make_naive_dd_settings(source, data_fixture, fields, curr_n, tgt_n):
    return {"id": "data_sha1", "unique_fields": fields}


deduper_test_params = [("naive", NaiveDeduper, make_naive_dd_settings)]


@pytest.mark.parametrize(
    "source, data_fixture, fields, curr_n, tgt_n", data_test_params
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
    curr_n,
    tgt_n,
    # Methodology params
    deduper_name,
    deduper_class,
    build_deduper_settings,
    # Pytest
    request,
):
    """Runs all deduper methodologies over exemplar tables."""
    df = request.getfixturevalue(data_fixture)
    # Confirm current and target shape from extremely naive dedupe
    assert isinstance(df, DataFrame)
    assert df.shape[0] == curr_n

    deduper_settings = build_deduper_settings(
        source, data_fixture, fields, curr_n, tgt_n
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
    assert deduped_df.shape[0] == tgt_n
    for field in fields:
        assert deduped_df_with_source[field + "_x"].equals(
            deduped_df_with_source[field + "_y"]
        )

    deduped.to_cmf(engine=db_engine[1])

    with Session(db_engine[1]) as session:
        model = session.query(Models).filter_by(name=f"{deduper_name}_{source}").first()
        proposed_dedupes = model.proposes_dedupes

    assert len(proposed_dedupes) == tgt_n

    db_clear_models(db_engine)
