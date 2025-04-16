"""
https://github.com/alvassin/alembic-quickstart/blob/master/tests/migrations/test_stairway.py

These tests were inspired by those at the link above, and there are many
others there we could implement. They also have good ideas about creating temporary
randomly-named databases that we should probably be using for all the tests.
"""

import pytest
from alembic.command import downgrade, upgrade
from alembic.config import Config
from alembic.script import Script, ScriptDirectory
from sqlalchemy import create_engine


@pytest.fixture()
def postgres_engine(alembic_config):
    """SQLAlchemy engine."""
    url = alembic_config.get_option("sqlalchemy.url")
    engine = create_engine(url, echo=True)
    try:
        yield engine
    finally:
        engine.dispose()


def alembic_config() -> Config:
    """Alembic config fixture."""
    return Config("test/fixtures/test-fixture-alembic.ini")


def get_revisions():
    config = alembic_config()
    revisions_dir = ScriptDirectory.from_config(config)
    revisions = list(revisions_dir.walk_revisions("base", "heads"))
    revisions.reverse()
    return revisions


@pytest.mark.parametrize("revision", get_revisions())
def test_migrations_stairway(alembic_config: Config, revision: Script):
    upgrade(alembic_config, revision.revision)
    downgrade(alembic_config, revision.down_revision or "-1")
    upgrade(alembic_config, revision.revision)
