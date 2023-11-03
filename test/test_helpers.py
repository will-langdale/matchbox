from cmf.data import Table
from cmf.helpers import selector, selectors

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
