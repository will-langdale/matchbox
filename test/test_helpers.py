import logging
import os

from dotenv import find_dotenv, load_dotenv
from matplotlib.figure import Figure
from pandas import DataFrame

from cmf import process, query
from cmf.clean import company_name, company_number
from cmf.helpers import (
    cleaner,
    cleaners,
    comparison,
    draw_model_tree,
    selector,
    selectors,
)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)


def test_selectors(db_engine):
    select_crn = selector(
        table=f"{os.getenv('SCHEMA')}.crn", fields=["id", "crn"], engine=db_engine[1]
    )
    select_duns = selector(
        table=f"{os.getenv('SCHEMA')}.duns", fields=["id", "duns"], engine=db_engine[1]
    )
    select_crn_duns = selectors(select_crn, select_duns)

    assert select_crn_duns is not None


def test_single_table_no_model_query(db_engine):
    """Tests query() on a single table. No point of truth to derive clusters"""
    select_crn = selector(
        table=f"{os.getenv('SCHEMA')}.crn", fields=["id", "crn"], engine=db_engine[1]
    )

    df_crn_sample = query(
        selector=select_crn,
        model=None,
        return_type="pandas",
        engine=db_engine[1],
        limit=10,
    )

    assert isinstance(df_crn_sample, DataFrame)
    assert df_crn_sample.shape[0] == 10

    df_crn_full = query(
        selector=select_crn, model=None, return_type="pandas", engine=db_engine[1]
    )

    assert df_crn_full.shape[0] == 3000
    assert set(df_crn_full.columns) == {
        "data_sha1",
        f"{os.getenv('SCHEMA')}_crn_id",
        f"{os.getenv('SCHEMA')}_crn_crn",
    }


def test_multi_table_no_model_query(db_engine):
    """Tests query() on multiple tables. No point of truth to derive clusters"""
    select_crn = selector(
        table=f"{os.getenv('SCHEMA')}.crn", fields=["id", "crn"], engine=db_engine[1]
    )
    select_duns = selector(
        table=f"{os.getenv('SCHEMA')}.duns", fields=["id", "duns"], engine=db_engine[1]
    )
    select_crn_duns = selectors(select_crn, select_duns)

    df_crn_duns_full = query(
        selector=select_crn_duns, model=None, return_type="pandas", engine=db_engine[1]
    )

    assert df_crn_duns_full.shape[0] == 3500
    assert (
        df_crn_duns_full[
            df_crn_duns_full[f"{os.getenv('SCHEMA')}_duns_id"].notnull()
        ].shape[0]
        == 500
    )
    assert (
        df_crn_duns_full[
            df_crn_duns_full[f"{os.getenv('SCHEMA')}_crn_id"].notnull()
        ].shape[0]
        == 3000
    )

    assert set(df_crn_duns_full.columns) == {
        "data_sha1",
        f"{os.getenv('SCHEMA')}_crn_id",
        f"{os.getenv('SCHEMA')}_crn_crn",
        f"{os.getenv('SCHEMA')}_duns_id",
        f"{os.getenv('SCHEMA')}_duns_duns",
    }


def test_single_table_with_model_query(
    db_engine, db_clear_models, db_add_dedupe_models, request
):
    """Tests query() on a single table using a model point of truth."""
    # Ensure database is clean, insert deduplicated models

    db_clear_models(db_engine)
    db_add_dedupe_models(db_engine, request)

    # Query

    select_crn = selector(
        table=f"{os.getenv('SCHEMA')}.crn",
        fields=["crn", "company_name"],
        engine=db_engine[1],
    )

    crn = query(
        selector=select_crn,
        model=f"naive_{os.getenv('SCHEMA')}.crn",
        return_type="pandas",
        engine=db_engine[1],
    )

    assert isinstance(crn, DataFrame)
    assert crn.shape[0] == 3000
    assert set(crn.columns) == {
        "cluster_sha1",
        "data_sha1",
        f"{os.getenv('SCHEMA')}_crn_crn",
        f"{os.getenv('SCHEMA')}_crn_company_name",
    }
    assert crn.cluster_sha1.nunique() == 1000


def test_multi_table_with_model_query(db_engine):
    """Tests query() on multiple tables using a model point of truth

    TODO: Implement. Will be a LOT easier to write when I have dedupers and
    linkers to generate data to query on -- not part of this MR.

    """
    # select_crn = selector(
    #     table=f"{os.getenv('SCHEMA')}.crn",
    #     fields=["id", "crn"],
    #     engine=db_engine[1]
    # )
    # select_duns = selector(
    #     table=f"{os.getenv('SCHEMA')}.duns",
    #     fields=["id", "duns"],
    #     engine=db_engine[1]
    # )
    # select_crn_duns = selectors(select_crn, select_duns)

    # df_crn_duns_full = query(
    #     selector=select_crn_duns,
    #     model="dd_m1",
    #     return_type="pandas",
    #     engine=db_engine[1]
    # )
    pass


def test_cleaners():
    cleaner_name = cleaner(function=company_name, arguments={"column": "company_name"})
    cleaner_number = cleaner(
        function=company_number, arguments={"column": "company_number"}
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    assert cleaner_name_number is not None


def test_process(db_engine):
    select_name = selector(
        table=f"{os.getenv('SCHEMA')}.crn",
        fields=["crn", "company_name"],
        engine=db_engine[1],
    )

    df_name = query(
        selector=select_name, model=None, return_type="pandas", engine=db_engine[1]
    )

    cleaner_name = cleaner(
        function=company_name,
        arguments={"column": f"{os.getenv('SCHEMA')}_crn_company_name"},
    )
    cleaner_number = cleaner(
        function=company_number, arguments={"column": f"{os.getenv('SCHEMA')}_crn_crn"}
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    df_name_cleaned = process(data=df_name, pipeline=cleaner_name_number)

    assert isinstance(df_name_cleaned, DataFrame)
    assert df_name_cleaned.shape[0] == 3000


def test_comparisons():
    comparison_name_id = comparison(
        sql_condition=(
            "l.company_name = r.company_name" " and l.data_hub_id = r.data_hub_id"
        )
    )

    assert comparison_name_id is not None


def test_draw_model_tree(db_engine):
    plt = draw_model_tree(db_engine[1])
    assert isinstance(plt, Figure)
