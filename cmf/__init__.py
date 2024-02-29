from os import environ

from dotenv import find_dotenv, load_dotenv

from cmf.data.results import to_clusters
from cmf.dedupers.make_deduper import make_deduper
from cmf.helpers.cleaner import process
from cmf.helpers.selector import query
from cmf.linkers.make_linker import make_linker

__all__ = ("make_deduper", "make_linker", "to_clusters", "process", "query")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

print(dotenv_path)

if "SCHEMA" not in environ:
    raise KeyError("SCHEMA environment variable not set.")
