import logging
import os

import pytest
import testing.postgresql
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy.schema import CreateSchema

from cmf.data import CMFBase

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)

CMF_POSTGRES = testing.postgresql.PostgresqlFactory(cache_initialized_db=True)


@pytest.fixture()
def db_engine():
    """
    Yield engine to mock in-memory database.
    """
    postgresql = CMF_POSTGRES()
    engine = create_engine(postgresql.url(), connect_args={"sslmode": "disable"})

    with engine.connect() as conn:
        # Create CMF schema
        if not inspect(conn).has_schema(os.getenv("SCHEMA")):
            conn.execute(CreateSchema(os.getenv("SCHEMA")))
            conn.commit()

        # Create CMF tables
        CMFBase.metadata.create_all(conn)
        conn.commit()

        LOGGER.info("Created in-memory CMF database")

    return postgresql, engine


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup the PostgreSQL database when we're done."""

    def teardown():
        CMF_POSTGRES.clear_cache()

    request.addfinalizer(teardown)
