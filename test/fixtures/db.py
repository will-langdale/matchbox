import hashlib
import logging
import os
import random

import pytest
import testing.postgresql
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema

from cmf.admin import add_dataset
from cmf.data import (
    Clusters,
    CMFBase,
    DDupeContains,
    DDupeProbabilities,
    Dedupes,
    LinkContains,
    LinkProbabilities,
    Links,
    Models,
    ModelsFrom,
    SourceData,
    SourceDataset,
    clusters_association,
)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)

CMF_POSTGRES = testing.postgresql.PostgresqlFactory(cache_initialized_db=True)


@pytest.fixture
def db_clear_all():
    """
    Returns a function to clear the database.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_clear_all(db_engine):
        db_metadata = MetaData(schema=os.getenv("SCHEMA"))
        db_metadata.reflect(bind=db_engine[1])
        with Session(db_engine[1]) as session:
            for table in reversed(db_metadata.sorted_tables):
                LOGGER.info(f"{table}")
                session.execute(table.delete())
            session.commit()

    return _db_clear_all


@pytest.fixture
def db_clear_data():
    """
    Returns a function to clear the SourceDatasets and SourceData tables.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_clear_data(db_engine):
        with Session(db_engine[1]) as session:
            session.query(SourceData).delete()
            session.query(SourceDataset).delete()
            session.commit()

    return _db_clear_data


@pytest.fixture
def db_clear_models():
    """
    Returns a function to clear the Models, ModelsFrom, Dedupes,
    DDupeProbabilities, DDupeContains, Links, LinkProbabilities,
    LinkContains and Clusterstables.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_clear_models(db_engine):
        with Session(db_engine[1]) as session:
            session.query(LinkProbabilities).delete()
            session.query(Links).delete()
            session.query(DDupeProbabilities).delete()
            session.query(Dedupes).delete()
            session.query(DDupeContains).delete()
            session.query(LinkContains).delete()
            session.query(clusters_association).delete()
            session.query(Clusters).delete()
            session.commit()

            session.query(ModelsFrom).delete()
            session.query(Models).delete()
            session.commit()

    return _db_clear_models


@pytest.fixture(scope="session")
def db_add_data(crn_companies, duns_companies, cdms_companies):
    """
    Returns a function to add source data to the database.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_add_data(db_engine):
        with db_engine[1].connect() as conn:
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

    return _db_add_data


@pytest.fixture(scope="session")
def db_add_models():
    """
    Returns a function to add models, clusters and probabilities data to the
    database.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_add_models(db_engine):
        with Session(db_engine[1]) as session:
            # Two Dedupers and two Linkers
            dd_m1 = Models(
                sha1=hashlib.sha1("dd_m1".encode()).digest(),
                name="dd_m1",
                description="",
            )
            dd_m2 = Models(
                sha1=hashlib.sha1("dd_m2".encode()).digest(),
                name="dd_m2",
                description="",
            )
            session.add_all([dd_m1, dd_m2])

            l_m1 = Models(
                sha1=hashlib.sha1("l_m1".encode()).digest(), name="l_m1", description=""
            )
            l_m2 = Models(
                sha1=hashlib.sha1("l_m2".encode()).digest(), name="l_m2", description=""
            )
            session.add_all([l_m1, l_m2])

            session.add_all(
                [
                    ModelsFrom(l_m1, dd_m1),
                    ModelsFrom(l_m1, dd_m2),
                    ModelsFrom(l_m2, dd_m1),
                    ModelsFrom(l_m2, dd_m2),
                ]
            )

            # Data, Dedupes and DDupeProbabilities
            data = session.query(SourceData).limit(6).all()

            dedupes = []
            for d1 in data:
                for d2 in data:
                    dedupes.append(
                        Dedupes(
                            sha1=hashlib.sha1(d1.sha1 + d2.sha1).digest(),
                            left=d1.sha1,
                            right=d2.sha1,
                        )
                    )

            ddupe_probs = []
            for dd in dedupes:
                ddupe_probs.append(
                    DDupeProbabilities(
                        ddupe=dd.sha1,
                        model=dd_m1.sha1,
                        probability=round(random.uniform(0, 1), 1),
                    )
                )
                ddupe_probs.append(
                    DDupeProbabilities(
                        ddupe=dd.sha1,
                        model=dd_m2.sha1,
                        probability=round(random.uniform(0, 1), 1),
                    )
                )
            session.add_all(dedupes)
            session.add_all(ddupe_probs)

            # Clusters, Links and LinkProbabilities
            clusters = [
                Clusters(sha1=hashlib.sha1(f"c{i}".encode()).digest()) for i in range(6)
            ]
            session.add_all(clusters)

            l_m1.creates = clusters
            l_m2.creates = clusters

            links = []
            for c1 in clusters:
                for c2 in clusters:
                    links.append(
                        Links(
                            sha1=hashlib.sha1(c1.sha1 + c2.sha1).digest(),
                            left=c1.sha1,
                            right=c2.sha1,
                        )
                    )
            link_probs = []
            for li in links:
                link_probs.append(
                    LinkProbabilities(
                        link=li.sha1,
                        model=l_m1.sha1,
                        probability=round(random.uniform(0, 1), 1),
                    )
                )
                link_probs.append(
                    LinkProbabilities(
                        link=li.sha1,
                        model=l_m2.sha1,
                        probability=round(random.uniform(0, 1), 1),
                    )
                )
            session.add_all(links)
            session.add_all(link_probs)

            session.commit()

    return _db_add_models


@pytest.fixture(scope="session")
def db_add_dedupe_models():
    """
    Returns a function to add Naive-deduplicated model probabilities and
    clusters to the database.

    Can be used to reset and repopulate between tests, when necessary.
    """
    pass


@pytest.fixture(scope="session")
def db_engine(db_add_data, db_add_models):
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
        db_add_data((postgresql, engine))

        # Insert models
        db_add_models((postgresql, engine))

    return postgresql, engine


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup the PostgreSQL database when we're done."""

    def teardown():
        CMF_POSTGRES.clear_cache()

    request.addfinalizer(teardown)
