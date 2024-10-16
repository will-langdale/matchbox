from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import (
    Engine,
    create_engine,
    text,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
)

from matchbox.server.base import MatchboxBackends, MatchboxSettings

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


class MatchboxPostgresCoreSettings(BaseModel):
    """Settings for Matchbox's PostgreSQL backend."""

    host: str
    port: int
    user: str
    password: str
    db_schema: str


class MatchboxPostgresSettings(MatchboxSettings):
    """Settings for the Matchbox PostgreSQL backend."""

    backend_type: MatchboxBackends = MatchboxBackends.POSTGRES

    postgres: MatchboxPostgresCoreSettings = Field(
        default_factory=MatchboxPostgresCoreSettings
    )


class MatchboxDatabase:
    def __init__(self, settings: MatchboxPostgresSettings):
        self.settings = settings
        self.engine: Engine | None = None
        self.SessionLocal: sessionmaker | None = None
        self.MatchboxBase = declarative_base()

    def connect(self):
        if not self.engine:
            connection_string = (
                f"postgresql://{self.settings.postgres.user}:{self.settings.postgres.password}"
                f"@{self.settings.postgres.host}:{self.settings.postgres.port}"
            )
            self.engine = create_engine(connection_string, logging_name="mb_pg_db")
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
            self.MatchboxBase.metadata.schema = self.settings.postgres.db_schema

    def get_engine(self) -> Engine:
        if not self.engine:
            self.connect()
        return self.engine

    def get_session(self):
        if not self.SessionLocal:
            self.connect()
        return self.SessionLocal()

    def create_database(self):
        self.connect()
        with self.engine.connect() as conn:
            conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {self.settings.postgres.db_schema};")
            )
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
            conn.commit()

        self.MatchboxBase.metadata.create_all(self.engine)


# Global database instance -- everything should use this

MBDB = MatchboxDatabase(MatchboxPostgresSettings())
