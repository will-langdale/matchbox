import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import MetaData, Table, insert, inspect
from sqlalchemy.orm import Session

from cmf.admin import add_dataset
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


def test_insert_data(db_engine, crn_companies, duns_companies, cdms_companies):
    """
    Test that when new data is added to the system, insertion updates it.
    Adds 5 rows to crn table, then adds the dataset again.
    Should result in the same as test_inserted_data + 5
    """
    new_data = [
        {
            "id": 3001,
            "company_name": "Twitterlist",
            "crn": "01HJ0TY5CRPT6ZWWJMH3K4DXH0",
        },
        {"id": 3002, "company_name": "Avaveo", "crn": "01HJ0TY5CR79KQT423SD5HMCXE"},
        {"id": 3003, "company_name": "Realmix", "crn": "01HJ0TY5CRRQBFQNVANJEPJ29D"},
        {"id": 3004, "company_name": "Eidel", "crn": "01HJ0TY5CRET0YPB0WF2R0DFEB"},
        {"id": 3005, "company_name": "Zoozzy", "crn": "01HJ0TY5CRHDX0NX5RSBJWSSKF"},
    ]
    with Session(db_engine[1]) as session:
        # Reflect the table and insert the data
        db_metadata = MetaData(schema=os.getenv("SCHEMA"))
        crn_table = Table(
            "crn",
            db_metadata,
            schema=os.getenv("SCHEMA"),
            autoload_with=session.get_bind(),
        )
        session.execute(insert(crn_table), new_data)
        session.commit()

        # Add the dataset again
        add_dataset(
            {
                "schema": os.getenv("SCHEMA"),
                "table": "crn",
                "id": "id",
            },
            db_engine[1],
        )

        # Test SourceData now contains 5 more rows
        inserted_rows = session.query(SourceData).count()
        raw_rows_plus5 = (
            crn_companies.drop(columns=["id"]).drop_duplicates().shape[0]
            + duns_companies.drop(columns=["id"]).drop_duplicates().shape[0]
            + cdms_companies.drop(columns=["id"]).drop_duplicates().shape[0]
            + 5
        )

    assert inserted_rows == raw_rows_plus5


def test_model_cluster_association(db_engine):
    """
    Test that cluster read/write via the association objects works as expected.

    Test that model deletion works as expected, removing the model and
    its creates edges in the association table, but not the clusters.
    """
    pass


def test_model_ddupe_association(db_engine):
    """
    Test that dedupe read/write via the association objects works as expected.

    Test that model deletion works as expected, removing the model and
    its proposes edges in the association object, but not the deduplications.
    """
    pass


def test_model_link_association(db_engine):
    """
    Test that link read/write via the association objects works as expected.

    Test that model deletion works as expected, removing the model and
    its proposes edges in the association object, but not the links.
    """
    pass
