"""Largely auto-generated Alembic configuration entrypoint."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import URL, engine_from_config, pool

from matchbox.server.postgresql.db import MBDB

config = context.config
fileConfig(config.config_file_name, disable_existing_loggers=False)

db_url = URL.create(
    "postgresql+psycopg",
    username=MBDB.settings.postgres.user,
    password=MBDB.settings.postgres.password,
    host=MBDB.settings.postgres.host,
    port=MBDB.settings.postgres.port,
    database=MBDB.settings.postgres.database,
)
config.set_main_option("sqlalchemy.url", db_url.render_as_string(hide_password=False))
target_metadata = MBDB.MatchboxBase.metadata


def run_migrations_online() -> None:
    """Run migrations based on SQLAlchemy ('offline' refers to raw SQL mode)."""
    connectable = engine_from_config(
        context.config.get_section(context.config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
        )

        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
