from os import environ

from dotenv import find_dotenv, load_dotenv

from matchbox.data.results import to_clusters
from matchbox.dedupers.make_deduper import make_deduper
from matchbox.helpers.cleaner import process
from matchbox.helpers.selector import query
from matchbox.linkers.make_linker import make_linker

__all__ = ("make_deduper", "make_linker", "to_clusters", "process", "query")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

if "MB_SCHEMA" not in environ:
    raise KeyError("MB_SCHEMA environment variable not set.")

if "MB_BATCH_SIZE" not in environ:
    raise KeyError("MB_BATCH_SIZE environment variable not set.")
