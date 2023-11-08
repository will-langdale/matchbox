from cmf.data import Table
from cmf.helpers import selector, selectors, cleaner, cleaners, comparison, comparisons
from cmf.clean import company_name, company_number

import os
from dotenv import load_dotenv, find_dotenv


def test_tables():
    load_dotenv(find_dotenv())

    fake_table = Table(db_schema=os.getenv("SCHEMA"), db_table="my_fake_table")
    real_table = Table(
        db_schema=os.getenv("SCHEMA"), db_table=os.getenv("PROBABILITIES_TABLE")
    )

    assert fake_table is not None
    assert real_table is not None


def test_selectors():
    select_dh = selector(
        table="dit.data_hub__companies", fields=["id", "company_number"]
    )
    select_ch = selector(
        table="companieshouse.companies", fields=["company_number", "company_name"]
    )
    select_dh_ch = selectors(select_dh, select_ch)

    assert select_dh_ch is not None


def test_cleaners():
    cleaner_name = cleaner(function=company_name, arguments={"column": "company_name"})
    cleaner_number = cleaner(
        function=company_number, arguments={"column": "company_number"}
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    assert cleaner_name_number is not None


def test_comparisons():
    comparison_name = comparison(sql_condition="company_name = company_name")
    comparison_id = comparison(sql_condition="data_hub_id = data_hub_id")
    comparison_name_id = comparisons(comparison_name, comparison_id)

    assert comparison_name_id is not None
