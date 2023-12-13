import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import testing.postgresql
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.schema import CreateSchema

import cmf.locations as loc
from cmf.admin import add_dataset
from cmf.data import CMFBase

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)

CMF_POSTGRES = testing.postgresql.PostgresqlFactory(cache_initialized_db=True)


@pytest.fixture(scope="session")
def all_companies():
    """
    Raw, correct company data.
    1,000 entries.
    """
    df = pd.read_csv(Path(loc.TEST, "data", "all_companies.csv")).reset_index(
        names="id"
    )
    return df


@pytest.fixture(scope="session")
def crn_companies(all_companies):
    """
    Company data split into CRN version.
    3,000 entries, 1,000 unique.
    """
    # Company name and CRN repeated 3 times
    df_crn = pd.DataFrame(
        np.repeat(all_companies.filter(["company_name", "crn"]).values, 3, axis=0)
    )
    df_crn.columns = ["company_name", "crn"]
    df_crn.reset_index(names="id", inplace=True)

    return df_crn


@pytest.fixture(scope="session")
def duns_companies(all_companies):
    """
    Company data split into DUNS version.
    500 entries.
    """
    # Company name and duns number, but only half
    df_duns = (
        all_companies.filter(["company_name", "duns"])
        .sample(frac=0.5)
        .reset_index(drop=True)
        .reset_index(names="id")
    )

    return df_duns


@pytest.fixture(scope="session")
def cdms_companies(all_companies):
    """
    Company data split into CDMS version.
    3,000 entries, 1,000 unique.
    """
    # CRN and CDMS refs repeated 3 times
    df_cdms = pd.DataFrame(
        np.repeat(all_companies.filter(["crn", "cdms"]).values, 3, axis=0)
    )
    df_cdms.columns = ["crn", "cdms"]
    df_cdms.reset_index(names="id", inplace=True)

    return df_cdms


@pytest.fixture(scope="session")
def db_engine(crn_companies, duns_companies, cdms_companies):
    """
    Yield engine to mock in-memory database.
    """
    postgresql = CMF_POSTGRES()
    engine = create_engine(postgresql.url(), connect_args={"sslmode": "disable"})

    with engine.connect() as conn:
        # Install relevant extensions
        conn.execute(text('create extension "uuid-ossp";'))
        conn.execute(text("create extension pgcrypto;"))
        conn.commit()

        # Create CMF schema
        if not inspect(conn).has_schema(os.getenv("SCHEMA")):
            conn.execute(CreateSchema(os.getenv("SCHEMA")))
            conn.commit()

        # Create CMF tables
        CMFBase.metadata.create_all(conn)
        conn.commit()

        LOGGER.info("Created in-memory CMF database")

        # Insert data
        crn_companies.to_sql(
            "crn",
            con=conn,
            schema=os.getenv("SCHEMA"),
            if_exists="replace",
            index=False,
        )
        duns_companies.to_sql(
            "duns",
            con=conn,
            schema=os.getenv("SCHEMA"),
            if_exists="replace",
            index=False,
        )
        cdms_companies.to_sql(
            "cdms",
            con=conn,
            schema=os.getenv("SCHEMA"),
            if_exists="replace",
            index=False,
        )

        LOGGER.info("Inserted raw data to database")

        datasets = {
            "crn_table": {
                "schema": os.getenv("SCHEMA"),
                "table": "crn",
                "id": "id",
            },
            "duns_table": {
                "schema": os.getenv("SCHEMA"),
                "table": "duns",
                "id": "id",
            },
            "cdms_table": {
                "schema": os.getenv("SCHEMA"),
                "table": "cdms",
                "id": "id",
            },
        }
        for dataset in datasets.values():
            add_dataset(dataset, conn)

        LOGGER.info("Raw data referenced in CMF")

    return postgresql, engine


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup the PostgreSQL database when we're done."""

    def teardown():
        CMF_POSTGRES.clear_cache()

    request.addfinalizer(teardown)
