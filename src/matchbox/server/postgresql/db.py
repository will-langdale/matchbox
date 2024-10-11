from dotenv import find_dotenv, load_dotenv
from sqlalchemy import Engine, MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, declared_attr, sessionmaker

from matchbox.common.exceptions import MatchboxConnectionError
from matchbox.server.postgresql.db import MatchboxPostgresSettings

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


class Base(DeclarativeBase):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    @classmethod
    def get_session(cls):
        return Session()


def connect_to_db(settings: MatchboxPostgresSettings) -> tuple[Base, Engine]:
    """Connect to Matchbox's Postgres backend database.

    Args:
        settings: The settings for Matchbox's PostgreSQL backend.

    Raises:
        MatchboxConnectionError: If the connection to the database fails.
    """
    schema = settings.schema

    mb_meta = MetaData(schema=schema)

    class MatchboxBase(Base):
        metadata = mb_meta

    engine = create_engine(
        f"postgresql://{settings.user}:{settings.password}@{settings.host}:{settings.port}/{settings.database}",
        logging_name="mb_pg_db",
    )

    global Session
    Session = sessionmaker(bind=engine)

    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
    except Exception as e:
        raise MatchboxConnectionError from e

    return MatchboxBase, engine
