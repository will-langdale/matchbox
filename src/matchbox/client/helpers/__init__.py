"""Core functionalities of the Matchbox client."""

from matchbox.client.helpers.comparison import comparison
from matchbox.client.helpers.delete import delete_resolution
from matchbox.client.helpers.index import get_source
from matchbox.client.helpers.selector import clean, select

__all__ = (
    # Sources
    "get_source",
    # Comparisons
    "comparison",
    # Deletion
    "delete_resolution",
    # Selectors
    "select",
    "clean",
)
