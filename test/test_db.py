import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from cmf.data import SourceData, SourceDataset

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def test_database(db_engine):
    """
    Test the database contains all the tables we expect.
    """
    tables = set(inspect(db_engine[1]).get_table_names(schema=os.getenv("SCHEMA")))
    to_check = {
        "crn",
        "duns",
        "cdms",
        "models_create_clusters",
        "clusters",
        "cluster_validation",
        "source_dataset",
        "source_data",
        "ddupes",
        "ddupe_probabilities",
        "ddupe_contains",
        "ddupe_validation",
        "links",
        "link_probabilities",
        "link_contains",
        "link_validation",
        "models",
        "models_from",
    }

    assert tables == to_check


def test_add_data(db_engine):
    """
    Test all datasets were inserted.
    """
    with Session(db_engine[1]) as session:
        inserted_tables = session.query(SourceDataset.db_table).all()
        inserted_tables = {t[0] for t in inserted_tables}
        expected_tables = {"crn", "duns", "cdms"}

    assert inserted_tables == expected_tables


def test_inserted_data(db_engine, crn_companies, duns_companies, cdms_companies):
    """
    Test all data was inserted.
    Note we drop duplicates because they're rolled up to arrays.
    """
    with Session(db_engine[1]) as session:
        inserted_rows = session.query(SourceData).count()
        raw_rows = (
            crn_companies.drop(columns=["id"]).drop_duplicates().shape[0]
            + duns_companies.drop(columns=["id"]).drop_duplicates().shape[0]
            + cdms_companies.drop(columns=["id"]).drop_duplicates().shape[0]
        )

    assert inserted_rows == raw_rows
