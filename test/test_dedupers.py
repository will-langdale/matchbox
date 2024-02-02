import os

import pytest
from pandas import DataFrame, Series, concat
from sqlalchemy.orm import Session

from cmf import make_deduper, process, query
from cmf.clean import company_name
from cmf.data import Models
from cmf.data import utils as du
from cmf.dedupers import Naive
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


# To parameterise

# Dedupers
# Name
# Class
# Settings

# Data
# Name
# Fields we care about
# Current count
# Target count

data_test_params = [
    (
        f"{os.getenv('SCHEMA')}.crn",
        "query_clean_crn",
        [f"{os.getenv('SCHEMA')}_crn_company_name", f"{os.getenv('SCHEMA')}_crn_crn"],
        3000,
        3000,  # every row is a duplicate, order doesn't count, so 3000
    )
]


@pytest.mark.parametrize(
    "source, data_fixture, fields, curr_n, tgt_n", data_test_params
)
def test_dedupers(db_engine, source, data_fixture, fields, curr_n, tgt_n, request):
    """Runs all deduper methodologies over exemplar tables."""
    df = request.getfixturevalue(data_fixture)
    # Confirm current and target shape from extremely naive dedupe
    assert isinstance(df, DataFrame)
    assert df.shape[0] == curr_n

    deduper = make_deduper(
        dedupe_run_name=source,
        description=f"Testing dedupe of {source}",
        deduper=Naive,
        deduper_settings={
            "id": "data_sha1",
            "unique_fields": fields,
        },
        data_source=source,
        data=df,
    )

    deduped = deduper()

    deduped_df = deduped.to_df()

    assert isinstance(deduped_df, DataFrame)
    assert deduped_df.shape[0] == tgt_n

    deduped.to_cmf(engine=db_engine[1])

    with Session(db_engine[1]) as session:
        model = session.query(Models).filter_by(name=source).first()
        proposed_dedupes = model.proposes_dedupes

    assert len(proposed_dedupes) == tgt_n  # successfully inserted 3000


# def test_naive_duns(db_engine, query_clean_duns):
#     """Dedupes a table made entirely of 500 unique items."""

#     col_prefix = f"{os.getenv('SCHEMA')}_duns_"
#     duns_naive_deduper = make_deduper(
#         dedupe_run_name="basic_duns",
#         description="Clean company name, DUNS",
#         deduper=Naive,
#         deduper_settings={
#             "id": f"{col_prefix}id",
#             "unique_fields": [f"{col_prefix}company_name", f"{col_prefix}duns"],
#         },
#         data_source=f"{os.getenv('SCHEMA')}.duns",
#         data=query_clean_duns,
#     )

#     duns_deduped = duns_naive_deduper()

#     duns_deduped_df = duns_deduped.to_df()

#     assert isinstance(duns_deduped_df, DataFrame)
#     assert duns_deduped_df.shape[0] == 0  # no duplicated rows

#     duns_deduped.to_cmf(engine=db_engine[1])

#     with Session(db_engine[1]) as session:
#         model = (
#             session.query(Models).filter_by(name="basic_duns").first()
#         )
#         proposed_dedupes = model.proposes_dedupes()

#     assert len(proposed_dedupes) == 0 # successfully inserted 0
