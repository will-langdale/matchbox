import hashlib
import logging
import os
import random
from typing import Callable, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema

from matchbox import make_deduper, make_linker, to_clusters
from matchbox.admin import add_dataset
from matchbox.data import (
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

from .models import DedupeTestParams, LinkTestParams, ModelTestParams

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def db_clear_all() -> Callable[[Engine], None]:
    """
    Returns a function to clear the database.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_clear_all(db_engine: Engine) -> None:
        db_metadata = MetaData(schema=os.getenv("SCHEMA"))
        db_metadata.reflect(bind=db_engine)
        with Session(db_engine) as session:
            for table in reversed(db_metadata.sorted_tables):
                LOGGER.info(f"{table}")
                session.execute(table.delete())
            session.commit()

    return _db_clear_all


@pytest.fixture
def db_clear_data() -> Callable[[Engine], None]:
    """
    Returns a function to clear the SourceDatasets and SourceData tables.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_clear_data(db_engine: Engine) -> None:
        with Session(db_engine) as session:
            session.query(SourceData).delete()
            session.query(SourceDataset).delete()
            session.commit()

    return _db_clear_data


@pytest.fixture
def db_clear_models() -> Callable[[Engine], None]:
    """
    Returns a function to clear the Models, ModelsFrom, Dedupes,
    DDupeProbabilities, DDupeContains, Links, LinkProbabilities,
    LinkContains and Clusterstables.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_clear_models(db_engine: Engine) -> None:
        with Session(db_engine) as session:
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
def db_add_data(
    crn_companies: DataFrame, duns_companies: DataFrame, cdms_companies: DataFrame
) -> Callable[[Engine], None]:
    """
    Returns a function to add source data to the database.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_add_data(db_engine: Engine) -> None:
        with db_engine.connect() as conn:
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
def db_add_models() -> Callable[[Engine], None]:
    """
    Returns a function to add models, clusters and probabilities data to the
    database.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_add_models(db_engine: Engine) -> None:
        with Session(db_engine) as session:
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

            dedupes_records = []
            for d1 in data:
                for d2 in data:
                    dedupes_records.append(
                        {
                            "sha1": hashlib.sha1(d1.sha1 + d2.sha1).digest(),
                            "left": d1.sha1,
                            "right": d2.sha1,
                        }
                    )

            dedupes = session.scalars(
                insert(Dedupes).returning(Dedupes), dedupes_records
            ).all()

            ddupe_probs_dd_m1 = []
            ddupe_probs_dd_m2 = []
            for dd in dedupes:
                p1 = DDupeProbabilities(probability=round(random.uniform(0, 1), 1))
                p1.dedupes = dd
                ddupe_probs_dd_m1.append(p1)

                p2 = DDupeProbabilities(probability=round(random.uniform(0, 1), 1))
                p2.dedupes = dd
                ddupe_probs_dd_m2.append(p2)

            dd_m1.proposes_dedupes.add_all(ddupe_probs_dd_m1)
            dd_m2.proposes_dedupes.add_all(ddupe_probs_dd_m2)

            # Clusters, Links and LinkProbabilities
            clusters_records = [
                {"sha1": hashlib.sha1(f"c{i}".encode()).digest()} for i in range(6)
            ]
            clusters = session.scalars(
                insert(Clusters).returning(Clusters), clusters_records
            ).all()

            l_m1.creates.add_all(clusters)
            l_m2.creates.add_all(clusters)

            link_records = []
            for c1 in clusters:
                for c2 in clusters:
                    link_records.append(
                        {
                            "sha1": hashlib.sha1(c1.sha1 + c2.sha1).digest(),
                            "left": c1.sha1,
                            "right": c2.sha1,
                        }
                    )

            links = session.scalars(insert(Links).returning(Links), link_records).all()

            link_probs_l_m1 = []
            link_probs_l_m2 = []
            for li in links:
                p1 = LinkProbabilities(probability=round(random.uniform(0, 1), 1))
                p1.links = li
                link_probs_l_m1.append(p1)

                p2 = LinkProbabilities(probability=round(random.uniform(0, 1), 1))
                p2.links = li
                link_probs_l_m2.append(p2)

            l_m1.proposes_links.add_all(link_probs_l_m1)
            l_m2.proposes_links.add_all(link_probs_l_m2)

            session.commit()

    return _db_add_models


@pytest.fixture(scope="session")
def db_add_dedupe_models_and_data() -> (
    Callable[
        [Engine, list[DedupeTestParams], list[ModelTestParams], FixtureRequest], None
    ]
):
    """
    Returns a function to add Naive-deduplicated model probabilities and
    clusters to the database.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_add_dedupe_models_and_data(
        db_engine: Engine,
        dedupe_data: list[DedupeTestParams],
        dedupe_models: list[ModelTestParams],
        request: FixtureRequest,
    ) -> None:
        for fx_data in dedupe_data:
            for fx_deduper in dedupe_models:
                df = request.getfixturevalue(fx_data.fixture)

                deduper_name = f"{fx_deduper.name}_{fx_data.source}"
                deduper_settings = fx_deduper.build_settings(fx_data)

                deduper = make_deduper(
                    dedupe_run_name=deduper_name,
                    description=(
                        f"Dedupe of {fx_data.source} " f"with {fx_deduper.name} method."
                    ),
                    deduper=fx_deduper.cls,
                    deduper_settings=deduper_settings,
                    data_source=fx_data.source,
                    data=df,
                )

                deduped = deduper()

                clustered = to_clusters(
                    df, results=deduped, key="data_sha1", threshold=0
                )

                deduped.to_cmf(engine=db_engine)
                clustered.to_cmf(engine=db_engine)

    return _db_add_dedupe_models_and_data


@pytest.fixture(scope="session")
def db_add_link_models_and_data() -> (
    Callable[
        [
            Engine,
            Callable[
                [Engine, list[DedupeTestParams], list[ModelTestParams], FixtureRequest],
                None,
            ],
            list[DedupeTestParams],
            list[ModelTestParams],
            list[LinkTestParams],
            list[ModelTestParams],
            FixtureRequest,
        ],
        None,
    ]
):
    """
    Returns a function to add Deterministic-linked model probabilities and
    clusters to the database.

    Can be used to reset and repopulate between tests, when necessary.
    """

    def _db_add_link_models_and_data(
        db_engine: Engine,
        db_add_dedupe_models_and_data: Callable[
            [Engine, list[DedupeTestParams], list[ModelTestParams], FixtureRequest],
            None,
        ],
        dedupe_data: list[DedupeTestParams],
        dedupe_models: list[ModelTestParams],
        link_data: list[LinkTestParams],
        link_models: list[ModelTestParams],
        request: FixtureRequest,
    ) -> None:
        db_add_dedupe_models_and_data(
            db_engine=db_engine,
            dedupe_data=dedupe_data,
            dedupe_models=dedupe_models,
            request=request,
        )

        for fx_data in link_data:
            for fx_linker in link_models:
                df_l = request.getfixturevalue(fx_data.fixture_l)
                df_r = request.getfixturevalue(fx_data.fixture_r)

                linker_name = f"{fx_linker.name}_{fx_data.source_l}_{fx_data.source_r}"
                linker_settings = fx_linker.build_settings(fx_data)

                linker = make_linker(
                    link_run_name=linker_name,
                    description=(
                        f"Testing link of {fx_data.source_l} and {fx_data.source_r} "
                        f"with {fx_linker.name} method."
                    ),
                    linker=fx_linker.cls,
                    linker_settings=linker_settings,
                    left_data=df_l,
                    left_source=fx_data.source_l,
                    right_data=df_r,
                    right_source=fx_data.source_r,
                )

                linked = linker()

                clustered = to_clusters(
                    df_l, df_r, results=linked, key="cluster_sha1", threshold=0
                )

                linked.to_cmf(engine=db_engine)
                clustered.to_cmf(engine=db_engine)

    return _db_add_link_models_and_data


@pytest.fixture(scope="session")
def db_engine(
    db_add_data: Callable[[Engine], None], db_add_models: Callable[[Engine], None]
) -> Generator[Engine, None, None]:
    """
    Yield engine to Docker container database.
    """
    load_dotenv(find_dotenv())
    engine = create_engine(
        url="postgresql://testuser:testpassword@localhost:5432/testdb",
        connect_args={"sslmode": "disable", "client_encoding": "utf8"},
    )

    with engine.connect() as conn:
        # Create CMF schema
        if not inspect(conn).has_schema(os.getenv("SCHEMA")):
            conn.execute(CreateSchema(os.getenv("SCHEMA")))
            conn.commit()

        # Create CMF tables
        CMFBase.metadata.create_all(conn)
        conn.commit()

        LOGGER.info("Created in-memory CMF database")

        # Insert data
        db_add_data(engine)

        # Insert models
        db_add_models(engine)

    yield engine
    engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def cleanup(db_engine, request):
    """Cleanup the PostgreSQL database by dropping all tables when we're done."""

    def teardown():
        with db_engine.connect() as conn:
            inspector = inspect(conn)
            for table_name in inspector.get_table_names(schema=os.getenv("SCHEMA")):
                conn.execute(
                    text(
                        f'DROP TABLE IF EXISTS "{os.getenv("SCHEMA")}".'
                        f'"{table_name}" CASCADE;'
                    )
                )
            conn.commit()

    request.addfinalizer(teardown)
