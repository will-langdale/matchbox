import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, TypeAlias

import boto3
import pyarrow as pa
import pytest
import respx
from _pytest.fixtures import FixtureRequest
from httpx import Client
from moto import mock_aws
from pandas import DataFrame
from respx import MockRouter
from sqlalchemy import Engine, create_engine
from sqlalchemy import text as sqltext

from matchbox import index, make_model
from matchbox.client._handler import create_client
from matchbox.client._settings import ClientSettings, settings
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.models import query_to_model_factory
from matchbox.common.factories.sources import linked_sources_factory
from matchbox.common.sources import Source, SourceAddress
from matchbox.server.base import (
    MatchboxDatastoreSettings,
    MatchboxDBAdapter,
    MatchboxSnapshot,
)
from matchbox.server.postgresql import MatchboxPostgres, MatchboxPostgresSettings

from .models import (
    DedupeTestParams,
    LinkTestParams,
    ModelTestParams,
    dedupe_data_test_params,
    dedupe_model_test_params,
    link_data_test_params,
    link_model_test_params,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


AddIndexedDataCallable = Callable[[MatchboxPostgres, list[Source]], None]
ScenarioCallable: TypeAlias = Callable[
    [MatchboxDBAdapter, Literal["index", "dedupe", "link"], int, int],
    Generator[TestkitDAG, None, None],
]


@pytest.fixture(scope="session")
def db_add_indexed_data() -> AddIndexedDataCallable:
    """Factory to create the indexing stage of matching."""

    def _db_add_indexed_data(
        backend: MatchboxDBAdapter,
        warehouse_data: list[Source],
    ):
        """Indexes data from the warehouse."""
        for source in warehouse_data:
            index(
                full_name=source.address.full_name,
                db_pk=source.db_pk,
                engine=source.engine,
                columns=[c.model_dump() for c in source.columns],
            )

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
                        f"Dedupe of {fx_data.source} with {fx_deduper.name} method."
                    ),
                    model_class=fx_deduper.cls,
                    model_settings=deduper_settings,
                    left_data=df,
                    left_resolution=fx_data.source,
                )

                results = model.run()
                results.to_matchbox()
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
                    left_resolution=fx_data.source_l,
                    right_data=df_r,
                    right_resolution=fx_data.source_r,
                )

                results = model.run()
                results.to_matchbox()
                model.truth = 0.0

    return _db_add_link_models_and_data


SetupDatabaseCallable = Callable[
    [MatchboxDBAdapter, list[Source], Literal["index", "dedupe", "link"]], None
]


# Global cache for database snapshots
_DATABASE_SNAPSHOTS = {}


def _generate_cache_key(
    backend: MatchboxDBAdapter,
    warehouse_data: list[Source],
    setup_level: Literal["index", "dedupe", "link"],
):
    """Generate a unique hash based on input parameters"""
    backend_key = backend.__class__.__name__

    data_str = json.dumps(
        [source.model_dump(exclude_unset=True) for source in warehouse_data],
        sort_keys=True,
    )

    key = f"{backend_key}_{data_str}_{setup_level}"
    return hashlib.md5(key.encode()).hexdigest()


@pytest.fixture(scope="function")
def setup_database(
    request: pytest.FixtureRequest,
) -> SetupDatabaseCallable:
    def _setup_database(
        backend: MatchboxDBAdapter,
        warehouse_data: list[Source],
        setup_level: Literal["index", "dedupe", "link"],
    ) -> None:
        # Check cache for existing snapshot
        cache_key = _generate_cache_key(backend, warehouse_data, setup_level)

        if cache_key in _DATABASE_SNAPSHOTS:
            backend.restore(clear=True, snapshot=_DATABASE_SNAPSHOTS[cache_key])
            return

        # Setup database
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

        # Save snapshot for future use
        _DATABASE_SNAPSHOTS[cache_key] = backend.dump()

    return _setup_database


def reconcile_engine_and_testkitdag(engine: Engine, dag: TestkitDAG) -> None:
    """Reconcile a TestkitDAG with a warehouse engine.

    * Writes all data to the warehouse, replacing existing data
    * Updates the engine of all sources in the DAG
    """
    for source_testkit in dag.sources.values():
        source_testkit.to_warehouse(engine)
        source_testkit.source.set_engine(engine)


def write_testkitdag_to_backend(backend: MatchboxDBAdapter, dag: TestkitDAG) -> None:
    """Write all sources and models from a TestkitDAG to a backend."""
    # Index sources
    for source_testkit in dag.sources.values():
        backend.index(
            source=source_testkit.source, data_hashes=source_testkit.data_hashes
        )

    # Add models in dependency order
    for model_testkit in sorted(
        dag.models.values(),
        key=lambda model_testkit: len(dag.adjacency[model_testkit.name]),
    ):
        backend.insert_model(model=model_testkit.model.metadata)
        backend.set_model_results(
            model=model_testkit.model.metadata.name, results=model_testkit.probabilities
        )


def create_scenario_dag(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    scenario_type: Literal["bare", "index", "dedupe", "link"],
    n_entities: int = 10,
    seed: int = 42,
) -> TestkitDAG:
    """Create a TestkitDAG representing a test scenario with backend integration.

    This approach first creates source data, writes it to backend and warehouse,
    then builds models by querying the backend to ensure ID alignment.
    """
    dag = TestkitDAG()

    # 1. Create linked sources
    linked = linked_sources_factory(n_true_entities=n_entities, seed=seed)
    dag.add_source(linked)

    # 2. Write sources to warehouse
    reconcile_engine_and_testkitdag(warehouse_engine, dag)

    # End here for bare database scenarios
    if scenario_type == "bare":
        return dag

    # 3. Index sources in backend
    for source_testkit in dag.sources.values():
        backend.index(
            source=source_testkit.source, data_hashes=source_testkit.data_hashes
        )

    # End here for index-only scenarios
    if scenario_type == "index":
        return dag

    # 4. Create and add deduplication models
    if scenario_type in ["dedupe", "link"]:
        # Build deduplication models
        for source_name in ["crn", "duns", "cdms"]:
            model_name = f"naive_test.{source_name}"

            # Query the raw data
            source_query = backend.query(
                source_address=linked.sources[source_name].source.address,
                resolution_name=source_name,
            )

            # Build model testkit using query data
            model_testkit = query_to_model_factory(
                left_resolution=source_name,
                left_query=source_query,
                left_source_pks={source_name: "source_pk"},
                true_entities=tuple(linked.true_entities.values()),
                name=model_name,
                description=f"Deduplication of {source_name}",
                seed=seed,
            )

            # Add to backend and DAG
            backend.insert_model(model=model_testkit.model.metadata)
            backend.set_model_results(
                model=model_name, results=model_testkit.probabilities
            )
            dag.add_model(model_testkit)

    # End here for dedupe-only scenarios
    if scenario_type == "dedupe":
        return dag

    # 5. Create linking models
    if scenario_type == "link":
        # First create CRN-DUNS link
        crn_model = dag.models["naive_test.crn"]
        duns_model = dag.models["naive_test.duns"]
        cdms_model = dag.models["naive_test.cdms"]

        # Query data for each resolution
        crn_query = backend.query(
            source_address=linked.sources["crn"].source.address,
            resolution_name=crn_model.name,
        )

        duns_query = backend.query(
            source_address=linked.sources["duns"].source.address,
            resolution_name=duns_model.name,
        )

        cdms_query = backend.query(
            source_address=linked.sources["cdms"].source.address,
            resolution_name=cdms_model.name,
        )

        # Create CRN-DUNS link
        crn_duns_name = "deterministic_naive_test.crn_naive_test.duns"
        crn_duns_model = query_to_model_factory(
            left_resolution=crn_model.name,
            left_query=crn_query,
            left_source_pks={"crn": "source_pk"},
            right_resolution=duns_model.name,
            right_query=duns_query,
            right_source_pks={"duns": "source_pk"},
            true_entities=tuple(linked.true_entities.values()),
            name=crn_duns_name,
            description="Link between CRN and DUNS",
            seed=seed,
        )

        # Add to backend and DAG
        backend.insert_model(model=crn_duns_model.model.metadata)
        backend.set_model_results(
            model=crn_duns_name, results=crn_duns_model.probabilities
        )
        dag.add_model(crn_duns_model)

        # Create CRN-CDMS link
        crn_cdms_name = "deterministic_naive_test.crn_naive_test.cdms"
        crn_cdms_model = query_to_model_factory(
            left_resolution=crn_model.name,
            left_query=crn_query,
            left_source_pks={"crn": "source_pk"},
            right_resolution=cdms_model.name,
            right_query=cdms_query,
            right_source_pks={"cdms": "source_pk"},
            true_entities=tuple(linked.true_entities.values()),
            name=crn_cdms_name,
            description="Link between CRN and CDMS",
            seed=seed,
        )

        backend.insert_model(model=crn_cdms_model.model.metadata)
        backend.set_model_results(
            model=crn_cdms_name, results=crn_cdms_model.probabilities
        )
        dag.add_model(crn_cdms_model)

        # Create final join
        # Query the previous link's results
        crn_cdms_query_crn_only = backend.query(
            source_address=linked.sources["crn"].source.address,
            resolution_name=crn_cdms_name,
        ).rename_columns(["id", "source_pk_crn"])
        crn_cdms_query_cdms_only = backend.query(
            source_address=linked.sources["cdms"].source.address,
            resolution_name=crn_cdms_name,
        ).rename_columns(["id", "source_pk_cdms"])
        crn_cdms_query = pa.concat_tables(
            [crn_cdms_query_crn_only, crn_cdms_query_cdms_only],
            promote_options="default",
        ).combine_chunks()

        duns_query_linked = backend.query(
            source_address=linked.sources["duns"].source.address,
            resolution_name=crn_duns_name,
        )

        final_join_name = "final_join"
        final_join_model = query_to_model_factory(
            left_resolution=crn_cdms_name,
            left_query=crn_cdms_query,
            left_source_pks={"crn": "source_pk_crn", "cdms": "source_pk_cdms"},
            right_resolution=duns_model.name,
            right_query=duns_query_linked,
            right_source_pks={"duns": "source_pk"},
            true_entities=tuple(linked.true_entities.values()),
            name=final_join_name,
            description="Final join of all entities",
            seed=seed,
        )

        backend.insert_model(model=final_join_model.model.metadata)
        backend.set_model_results(
            model=final_join_name, results=final_join_model.probabilities
        )
        dag.add_model(final_join_model)

    return dag


_DATABASE_SNAPSHOTS_CACHE: dict[str, tuple[TestkitDAG, MatchboxSnapshot]] = {}


@pytest.fixture(scope="function")
def scenario(sqlite_warehouse: Engine) -> ScenarioCallable:
    """Fixture that provides a TestkitDAG with appropriate database setup."""

    @contextmanager
    def _scenario(
        backend: MatchboxDBAdapter,
        scenario_type: Literal["bare", "index", "dedupe", "link"],
        n_entities: int = 10,
        seed: int = 42,
    ) -> Generator[TestkitDAG, None, None]:
        """Context manager for creating TestkitDAG scenarios."""
        # Generate cache key for backend snapshot
        cache_key = f"{backend.__class__.__name__}_{scenario_type}_{n_entities}_{seed}"

        # Check if we have a backend snapshot cached
        if cache_key in _DATABASE_SNAPSHOTS_CACHE:
            # Load cached snapshot and DAG
            dag, snapshot = _DATABASE_SNAPSHOTS_CACHE[cache_key]
            dag = dag.model_copy(deep=True)

            # Restore backend and write sources to warehouse
            backend.restore(clear=True, snapshot=snapshot)
            reconcile_engine_and_testkitdag(sqlite_warehouse, dag)
        else:
            # Create new TestkitDAG with proper backend integration
            dag = create_scenario_dag(
                backend, sqlite_warehouse, scenario_type, n_entities, seed
            )

            # Cache the snapshot and DAG
            _DATABASE_SNAPSHOTS_CACHE[cache_key] = (dag, backend.dump())

        yield dag

        backend.clear(certain=True)

    return _scenario


# Warehouse database fixtures


@pytest.fixture(scope="session")
def warehouse_engine() -> Engine:
    """Creates an engine for the test warehouse database"""
    user = "warehouse_user"
    password = "warehouse_password"
    host = "localhost"
    database = "warehouse"
    port = 7654
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")


@pytest.fixture(scope="session")
def warehouse_data(
    warehouse_engine: Engine,
    crn_companies: DataFrame,
    duns_companies: DataFrame,
    cdms_companies: DataFrame,
) -> Generator[list[Source], None, None]:
    """Inserts data into the warehouse database for testing."""
    with warehouse_engine.connect() as conn:
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

    with warehouse_engine.connect() as conn:
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
        Source(address=SourceAddress.compose(warehouse_engine, "test.crn"), db_pk="id")
        .set_engine(warehouse_engine)
        .default_columns(),
        Source(address=SourceAddress.compose(warehouse_engine, "test.duns"), db_pk="id")
        .set_engine(warehouse_engine)
        .default_columns(),
        Source(address=SourceAddress.compose(warehouse_engine, "test.cdms"), db_pk="id")
        .set_engine(warehouse_engine)
        .default_columns(),
    ]

    # Clean up the warehouse data
    with warehouse_engine.connect() as conn:
        conn.execute(sqltext("drop table if exists test.crn;"))
        conn.execute(sqltext("drop table if exists test.duns;"))
        conn.execute(sqltext("drop table if exists test.cdms;"))
        conn.commit()


@pytest.fixture(scope="function")
def sqlite_warehouse() -> Generator[Engine, None, None]:
    """Creates an engine for a function-scoped SQLite warehouse database.

    By using a temporary file, produces a URI that can be shared between processes.
    """
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        yield engine
        engine.dispose()


# Matchbox database fixtures


@pytest.fixture(scope="session")
def matchbox_datastore() -> MatchboxDatastoreSettings:
    """Settings for the Matchbox datastore."""
    return MatchboxDatastoreSettings(
        host="localhost",
        port=9000,
        access_key_id="access_key_id",
        secret_access_key="secret_access_key",
        default_region="eu-west-2",
        cache_bucket_name="cache",
    )


@pytest.fixture(scope="session")
def matchbox_settings(
    matchbox_datastore: MatchboxDatastoreSettings,
) -> MatchboxPostgresSettings:
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
        datastore=matchbox_datastore,
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


# Mock AWS fixtures


@pytest.fixture(scope="function")
def aws_credentials() -> None:
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "eu-west-2"


@pytest.fixture(scope="function")
def s3(aws_credentials: None) -> Generator[S3Client, None, None]:
    """Return a mocked S3 client."""
    with mock_aws():
        yield boto3.client("s3", region_name="eu-west-2")


# API, mocked and Docker


@pytest.fixture(scope="function")
def matchbox_api() -> Generator[MockRouter, None, None]:
    """Client for the mocked Matchbox API."""
    with respx.mock(base_url=settings.api_root, assert_all_called=True) as respx_mock:
        yield respx_mock


@pytest.fixture(scope="session")
def matchbox_client_settings() -> ClientSettings:
    """Client settings for the Matchbox API running in Docker."""
    return settings


@pytest.fixture(scope="session")
def matchbox_client(matchbox_client_settings: ClientSettings) -> Client:
    """Client for the Matchbox API running in Docker."""
    return create_client(settings=matchbox_client_settings)
