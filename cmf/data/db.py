import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, declarative_base

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

if "SCHEMA" not in os.environ:
    raise KeyError("SCHEMA environment variable not set.")


def connect_to_db(schema: str = os.environ["SCHEMA"]) -> DeclarativeBase:
    cmf_meta = MetaData(schema=os.environ["SCHEMA"])
    CMFBase = declarative_base(metadata=cmf_meta)
    return CMFBase


CMFBase = connect_to_db(schema=os.environ["SCHEMA"])

ENGINE = create_engine("postgresql://", logging_name="cmf_db")
