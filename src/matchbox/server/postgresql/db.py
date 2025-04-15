"""Matchbox PostgreSQL database connection."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from adbc_driver_postgresql import dbapi as adbc_dbapi
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from pydantic import BaseModel, Field
from sqlalchemy import Engine, MetaData, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

from matchbox.common.logging import logger
from matchbox.server.base import MatchboxBackends, MatchboxServerSettings

ALEMBIC_INI_PATH = (Path(__file__).parents[4] / "alembic.ini").resolve()


alembic_cfg = Config(ALEMBIC_INI_PATH)


class MatchboxPostgresCoreSettings(BaseModel):
    """PostgreSQL-specific settings for Matchbox."""

    host: str
    port: int
    user: str
    password: str
    database: str
    db_schema: str


class MatchboxPostgresSettings(MatchboxServerSettings):
    """Settings for the Matchbox PostgreSQL backend.

    Inherits the core settings and adds the PostgreSQL-specific settings.
    """

    backend_type: MatchboxBackends = MatchboxBackends.POSTGRES

    postgres: MatchboxPostgresCoreSettings = Field(
        default_factory=MatchboxPostgresCoreSettings
    )


class MatchboxDatabase:
    """Matchbox PostgreSQL database connection."""

    def __init__(self, settings: MatchboxPostgresSettings):
        """Initialise the database connection."""
        self.settings = settings
        self._engine: Engine | None = None
        self._SessionLocal: sessionmaker | None = None
        self._adbc_pool: QueuePool | None = None
        self._source_adbc_connection: adbc_dbapi.Connection | None = None
        self.MatchboxBase = declarative_base(
            metadata=MetaData(schema=settings.postgres.db_schema)
        )

    def connection_string(self, driver: bool = True) -> str:
        """Get the connection string for PostgreSQL."""
        driver_string = ""
        if driver:
            driver_string = "+psycopg"
        return (
            f"postgresql{driver_string}://{self.settings.postgres.user}:{self.settings.postgres.password}"
            f"@{self.settings.postgres.host}:{self.settings.postgres.port}/"
            f"{self.settings.postgres.database}"
        )

    def _connect(self):
        """Connect to the database."""
        self._engine = create_engine(
            url=self.connection_string(), logging_name="matchbox.engine", echo=False
        )
        self._SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

    def _disconnect(self):
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._SessionLocal = None

    def _connect_adbc(self) -> None:
        self._source_adbc_connection = adbc_dbapi.connect(
            self.connection_string(driver=False)
        )
        self._adbc_pool = QueuePool(
            self._source_adbc_connection.adbc_clone,
        )

    def _disconnect_adbc(self) -> None:
        if self._adbc_pool:
            self._adbc_pool.dispose()
            self._adbc_pool = None

            self._source_adbc_connection.close()
            self._source_adbc_connection = None

    def _reset_connections(self) -> None:
        """Dispose and re-initialise SQLAlchemy and ADBC connection managers."""
        self._disconnect()
        self._connect()

        self._disconnect_adbc()
        self._connect_adbc()

    def get_engine(self) -> Engine:
        """Get the database engine."""
        if not self._engine:
            self._connect()
        return self._engine

    def get_session(self):
        """Get a new session."""
        if not self._SessionLocal:
            self._connect()
        return self._SessionLocal()

    @contextmanager
    def get_adbc_connection(self) -> Generator[adbc_dbapi.Connection, Any, Any]:
        """Get a new ADBC connection.

        The connection must be used within a context manager.
        """
        if not self._adbc_pool:
            self._connect_adbc()

        conn = self._adbc_pool.connect()
        try:
            yield conn.driver_connection
        finally:
            conn.close()

    def create_database(self):
        """Create the database and all tables expected in the schema."""
        # Check if schema exists, create if not
        command.upgrade(alembic_cfg, "head")

    def clear_database(self):
        """Delete all rows in every table in the database schema."""
        with self.get_engine().connect() as conn:
            for table in self.MatchboxBase.metadata.sorted_tables:
                conn.execute(table.delete())
            conn.commit()

    def drop_database(self):
        """Drop all tables in the database schema and re-recreate them."""
        command.downgrade(alembic_cfg, "base")
        self._reset_connections()
        self.create_database()

    def verify_schema(self):
        """Verify the database schema live is in sync with the ORM.

        If any differences are detected, log this as an error.

        NOTE: this was originally implemented prior to alembic. In principle alembic
        is best placed to manage any such diff, and this remains for now only as an
        informative aid and could be removed.
        """
        engine = self.get_engine()

        with engine.connect() as conn:
            schemas = conn.dialect.get_schema_names(conn)
            if self.settings.postgres.db_schema not in schemas:
                self.create_database()

        # Compare schema with ORM
        def _include_name(name: str, type_: str, _: dict[str, str]) -> bool:
            if type_ == "schema":
                return name == self.settings.postgres.db_schema
            else:
                return True

        with engine.connect() as conn:
            opts = {
                "compare_type": True,
                "compare_server_default": True,
                "include_schemas": True,
                "include_names": _include_name,
            }
            context = MigrationContext.configure(conn, opts=opts)

            diff = compare_metadata(context, self.MatchboxBase.metadata)

            if diff:
                logger.warning(f"Schema mismatch detected. \nDiff: {diff}")
            else:
                logger.info("Schema matches expected.")


# Global database instance -- everything should use this

MBDB = MatchboxDatabase(MatchboxPostgresSettings())
