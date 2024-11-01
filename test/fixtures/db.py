from typing import Callable, Generator, Literal

import pytest
from _pytest.fixtures import FixtureRequest
from dotenv import find_dotenv, load_dotenv
from matchbox import make_model
from matchbox.server.base import MatchboxDBAdapter
from matchbox.server.models import Source, SourceWarehouse
from matchbox.server.postgresql import MatchboxPostgres, MatchboxPostgresSettings
from pandas import DataFrame
from sqlalchemy import text as sqltext

from .models import (
    DedupeTestParams,
    LinkTestParams,
    ModelTestParams,
    dedupe_data_test_params,
    dedupe_model_test_params,
    link_data_test_params,
    link_model_test_params,
)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

AddIndexedDataCallable = Callable[[MatchboxPostgres, list[Source]], None]


@pytest.fixture(scope="session")
def db_add_indexed_data() -> AddIndexedDataCallable:
    """Factory to create the indexing stage of matching."""

    def _db_add_indexed_data(
        backend: MatchboxDBAdapter,
        warehouse_data: list[Source],
    ):
        """Indexes data from the warehouse."""
        for dataset in warehouse_data:
            backend.index(dataset=dataset)

    return _db_add_indexed_data


AddDedupeModelsAndDataCallable = Callable[
    [
        AddIndexedDataCallable,
        MatchboxPostgres,
        list[Source],
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
        backend: MatchboxDBAdapter,
        warehouse_data: list[Source],
        dedupe_data: list[DedupeTestParams],
        dedupe_models: list[ModelTestParams],
        request: FixtureRequest,
    ) -> None:
        """Deduplicates data from the warehouse and logs in Matchbox."""
        db_add_indexed_data(backend=backend, warehouse_data=warehouse_data)

        for fx_data in dedupe_data:
            for fx_deduper in dedupe_models:
                _, df = request.getfixturevalue(fx_data.fixture)

                deduper_name = f"{fx_deduper.name}_{fx_data.source}"
                deduper_settings = fx_deduper.build_settings(fx_data)

                model = make_model(
                    model_name=deduper_name,
                    description=(
                        f"Dedupe of {fx_data.source} " f"with {fx_deduper.name} method."
                    ),
                    model_class=fx_deduper.cls,
                    model_settings=deduper_settings,
                    left_data=df,
                    left_source=fx_data.source,
                )

                results = model.run()
                results.to_matchbox(backend=backend)
                model.truth = 0.0

    return _db_add_dedupe_models_and_data


AddLinkModelsAndDataCallable = Callable[
    [
        AddIndexedDataCallable,
        AddDedupeModelsAndDataCallable,
        MatchboxPostgres,
        list[Source],
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
        backend: MatchboxDBAdapter,
        warehouse_data: list[Source],
        dedupe_data: list[DedupeTestParams],
        dedupe_models: list[ModelTestParams],
        link_data: list[LinkTestParams],
        link_models: list[ModelTestParams],
        request: FixtureRequest,
    ) -> None:
        """Links data from the warehouse and logs in Matchbox."""
        db_add_dedupe_models_and_data(
            db_add_indexed_data=db_add_indexed_data,
            backend=backend,
            warehouse_data=warehouse_data,
            dedupe_data=dedupe_data,
            dedupe_models=dedupe_models,
            request=request,
        )

        for fx_data in link_data:
            for fx_linker in link_models:
                _, df_l = request.getfixturevalue(fx_data.fixture_l)
                _, df_r = request.getfixturevalue(fx_data.fixture_r)

                linker_name = f"{fx_linker.name}_{fx_data.source_l}_{fx_data.source_r}"
                linker_settings = fx_linker.build_settings(fx_data)

                model = make_model(
                    model_name=linker_name,
                    description=(
                        f"Testing link of {fx_data.source_l} and {fx_data.source_r} "
                        f"with {fx_linker.name} method."
                    ),
                    model_class=fx_linker.cls,
                    model_settings=linker_settings,
                    left_data=df_l,
                    left_source=fx_data.source_l,
                    right_data=df_r,
                    right_source=fx_data.source_r,
                )

                results = model.run()
                results.to_matchbox(backend=backend)
                model.truth = 0.0

    return _db_add_link_models_and_data


SetupDatabaseCallable = Callable[
    [MatchboxDBAdapter, list[Source], Literal["index", "dedupe", "link"]], None
]


@pytest.fixture(scope="function")
def setup_database(
    request: pytest.FixtureRequest,
) -> SetupDatabaseCallable:
    def _setup_database(
        backend: MatchboxDBAdapter,
        warehouse_data: list[Source],
        setup_level: Literal["index", "dedupe", "link"],
    ) -> None:
        db_add_indexed_data = request.getfixturevalue("db_add_indexed_data")
        db_add_dedupe_models_and_data = request.getfixturevalue(
            "db_add_dedupe_models_and_data"
        )
        db_add_link_models_and_data = request.getfixturevalue(
            "db_add_link_models_and_data"
        )

        backend.clear(certain=True)

        if setup_level == "index":
            db_add_indexed_data(backend=backend, warehouse_data=warehouse_data)
        elif setup_level == "dedupe":
            db_add_dedupe_models_and_data(
                db_add_indexed_data=db_add_indexed_data,
                backend=backend,
                warehouse_data=warehouse_data,
                dedupe_data=dedupe_data_test_params,
                dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper
                request=request,
            )
        elif setup_level == "link":
            db_add_link_models_and_data(
                db_add_indexed_data=db_add_indexed_data,
                db_add_dedupe_models_and_data=db_add_dedupe_models_and_data,
                backend=backend,
                warehouse_data=warehouse_data,
                dedupe_data=dedupe_data_test_params,
                dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper
                link_data=link_data_test_params,
                link_models=[link_model_test_params[0]],  # Deterministic linker
                request=request,
            )
        else:
            raise ValueError(f"Invalid setup level: {setup_level}")

    return _setup_database


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
) -> Generator[list[Source], None, None]:
    """Inserts data into the warehouse database for testing."""
    with warehouse.engine.connect() as conn:
        conn.execute(sqltext("drop schema if exists test cascade;"))
        conn.execute(sqltext("create schema test;"))
        crn_companies.to_sql(
            name="crn",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )
        duns_companies.to_sql(
            name="duns",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )
        cdms_companies.to_sql(
            name="cdms",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )
        conn.commit()

    with warehouse.engine.connect() as conn:
        assert (
            conn.execute(sqltext("select count(*) from test.crn;")).scalar()
            == crn_companies.shape[0]
        )
        assert (
            conn.execute(sqltext("select count(*) from test.duns;")).scalar()
            == duns_companies.shape[0]
        )
        assert (
            conn.execute(sqltext("select count(*) from test.cdms;")).scalar()
            == cdms_companies.shape[0]
        )

    yield [
        Source(database=warehouse, db_pk="id", db_schema="test", db_table="crn"),
        Source(database=warehouse, db_pk="id", db_schema="test", db_table="duns"),
        Source(database=warehouse, db_pk="id", db_schema="test", db_table="cdms"),
    ]

    # Clean up the warehouse data
    with warehouse.engine.connect() as conn:
        conn.execute(sqltext("drop table if exists test.crn;"))
        conn.execute(sqltext("drop table if exists test.duns;"))
        conn.execute(sqltext("drop table if exists test.cdms;"))
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
            "db_schema": "mb",
        },
    )


@pytest.fixture(scope="function")
def matchbox_postgres(
    matchbox_settings: MatchboxPostgresSettings,
) -> Generator[MatchboxPostgres, None, None]:
    """The Matchbox PostgreSQL database."""

    adapter = MatchboxPostgres(settings=matchbox_settings)

    # Clean up the Matchbox database before each test, just in case
    adapter.clear(certain=True)

    yield adapter

    # Clean up the Matchbox database after each test
    adapter.clear(certain=True)
