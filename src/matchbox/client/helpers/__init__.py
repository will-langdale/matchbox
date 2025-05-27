"""Core functionalities of the Matchbox client."""

from matchbox.client.helpers.cleaner import cleaner, cleaners
from matchbox.client.helpers.comparison import comparison
from matchbox.client.helpers.delete import delete_resolution
from matchbox.client.helpers.selector import select

__all__ = (
    # Cleaners
    "cleaner",
    "cleaners",
    # Comparisons
    "comparison",
    # Deletion
    "delete_resolution",
    # Selectors
    "select",
)
