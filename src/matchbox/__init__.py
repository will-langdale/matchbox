from dotenv import find_dotenv, load_dotenv

from matchbox.client.helpers.cleaner import process
from matchbox.client.helpers.index import index
from matchbox.client.helpers.selector import match, query
from matchbox.client.models.models import make_model

__all__ = ("make_model", "to_clusters", "process", "query", "match", "index")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)
