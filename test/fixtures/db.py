from typing import Callable, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from dotenv import find_dotenv, load_dotenv
from matchbox import make_deduper, make_linker, to_clusters
from matchbox.server.base import IndexableDataset, SourceWarehouse
from matchbox.server.postgresql import MatchboxPostgres, MatchboxPostgresSettings
from pandas import DataFrame
from sqlalchemy import text

from .models import DedupeTestParams, LinkTestParams, ModelTestParams

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

AddIndexedDataCallable = Callable[[MatchboxPostgres, list[IndexableDataset]], None]


@pytest.fixture(scope="session")
def db_add_indexed_data() -> AddIndexedDataCallable:
    """Factory to create the indexing stage of matching."""

    def _db_add_indexed_data(
        matchbox_postgres: MatchboxPostgres,
        warehouse_data: list[IndexableDataset],
    ):
        """Indexes data from the warehouse."""
        for dataset in warehouse_data:
            matchbox_postgres.index(dataset=dataset)

    return _db_add_indexed_data


AddDedupeModelsAndDataCallable = Callable[
    [
        AddIndexedDataCallable,
        MatchboxPostgres,
        list[IndexableDataset],
        list[DedupeTestParams],
        list[ModelTestParams],
        FixtureRequest,
    ],
    None,
]


@pytest.fixture(scope="session")
def db_add_dedupe_models_and_data() -> AddDedupeModelsAndDataCallable:
    """Factory to create the deduplication stage of matching."""

    def _db_add_dedupe_models_and_data(
        db_add_indexed_data: AddIndexedDataCallable,
        matchbox_postgres: MatchboxPostgres,
        warehouse_data: list[IndexableDataset],
        dedupe_data: list[DedupeTestParams],
        dedupe_models: list[ModelTestParams],
        request: FixtureRequest,
    ) -> None:
        """Deduplicates data from the warehouse and logs in Matchbox."""
        db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

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

                deduped.to_matchbox(backend=matchbox_postgres)
                clustered.to_matchbox(backend=matchbox_postgres)

    return _db_add_dedupe_models_and_data


AddLinkModelsAndDataCallable = Callable[
    [
        AddIndexedDataCallable,
        AddDedupeModelsAndDataCallable,
        MatchboxPostgres,
        list[IndexableDataset],
        list[DedupeTestParams],
        list[ModelTestParams],
        list[LinkTestParams],
        list[ModelTestParams],
        FixtureRequest,
    ],
    None,
]


@pytest.fixture(scope="session")
def db_add_link_models_and_data() -> AddLinkModelsAndDataCallable:
    """Factory to create the link stage of matching."""

    def _db_add_link_models_and_data(
        db_add_indexed_data: AddIndexedDataCallable,
        db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
        matchbox_postgres: MatchboxPostgres,
        warehouse_data: list[IndexableDataset],
        dedupe_data: list[DedupeTestParams],
        dedupe_models: list[ModelTestParams],
        link_data: list[LinkTestParams],
        link_models: list[ModelTestParams],
        request: FixtureRequest,
    ) -> None:
        """Links data from the warehouse and logs in Matchbox."""
        db_add_dedupe_models_and_data(
            backend=matchbox_postgres,
            warehouse_data=warehouse_data,
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

                linked.to_matchbox(backend=matchbox_postgres)
                clustered.to_matchbox(backend=matchbox_postgres)

    return _db_add_link_models_and_data


# Warehouse database fixtures


@pytest.fixture(scope="session")
def warehouse() -> SourceWarehouse:
    """Create a connection to the test warehouse database."""
    warehouse = SourceWarehouse(
        alias="test_warehouse",
        db_type="postgresql",
        user="warehouse_user",
        password="warehouse_password",
        host="localhost",
        database="warehouse",
        port=7654,
    )
    assert warehouse.engine
    return warehouse


@pytest.fixture(scope="session")
def warehouse_data(
    warehouse: SourceWarehouse,
    crn_companies: DataFrame,
    duns_companies: DataFrame,
    cdms_companies: DataFrame,
) -> Generator[list[IndexableDataset], None, None]:
    """Inserts data into the warehouse database for testing."""
    with warehouse.engine.connect() as conn:
        conn.execute(text("drop schema if exists test cascade;"))
        conn.execute(text("create schema test;"))
        crn_companies.to_sql(
            "crn",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )
        duns_companies.to_sql(
            "duns",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )
        cdms_companies.to_sql(
            "cdms",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )

    yield [
        IndexableDataset(
            database=warehouse, db_pk="id", db_schema="test", db_table="crn"
        ),
        IndexableDataset(
            database=warehouse, db_pk="id", db_schema="test", db_table="duns"
        ),
        IndexableDataset(
            database=warehouse, db_pk="id", db_schema="test", db_table="cdms"
        ),
    ]

    # Clean up the warehouse data
    with warehouse.engine.connect() as conn:
        conn.execute(text("drop table if exists test.crn;"))
        conn.execute(text("drop table if exists test.duns;"))
        conn.execute(text("drop table if exists test.cdms;"))
        conn.commit()


# Matchbox database fixtures


@pytest.fixture(scope="session")
def matchbox_settings() -> MatchboxPostgresSettings:
    """Settings for the Matchbox database."""
    return MatchboxPostgresSettings(
        batch_size=250_000,
        postgres={
            "host": "localhost",
            "port": 5432,
            "user": "matchbox_user",
            "password": "matchbox_password",
            "database": "matchbox",
            "db_schema": "matchbox",
        },
    )


@pytest.fixture(scope="function")
def matchbox_postgres(
    matchbox_settings: MatchboxPostgresSettings,
) -> Generator[MatchboxPostgres, None, None]:
    """The Matchbox PostgreSQL database."""

    adapter = MatchboxPostgres(settings=matchbox_settings)

    yield adapter

    # Clean up the Matchbox database after each test
    adapter.clear(certain=True)
