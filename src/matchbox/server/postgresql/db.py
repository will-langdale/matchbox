import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import Engine, MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, declared_attr, sessionmaker

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

if "MB_SCHEMA" not in os.environ:
    raise KeyError("MB_SCHEMA environment variable not set.")


class Base(DeclarativeBase):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    @classmethod
    def get_session(cls):
        return Session()


def connect_to_db(schema: str | None) -> tuple[Base, Engine]:
    if not schema:
        schema = os.environ["MB_SCHEMA"]

    cmf_meta = MetaData(schema=schema)

    class MatchboxBase(Base):
        metadata = cmf_meta

    engine = create_engine("postgresql://", logging_name="mb_db")

    global Session
    Session = sessionmaker(bind=engine)

    return MatchboxBase, engine


MatchboxBase, ENGINE = connect_to_db(schema=os.environ["MB_SCHEMA"])
