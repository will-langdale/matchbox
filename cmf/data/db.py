import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import MetaData, create_engine
from sqlalchemy.ext.declarative import declarative_base

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

cmf_meta = MetaData(schema=os.getenv("SCHEMA"))
CMFBase = declarative_base(metadata=cmf_meta)

ENGINE = create_engine("postgresql://", echo=False)
