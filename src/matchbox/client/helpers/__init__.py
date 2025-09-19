"""Core functionalities of the Matchbox client."""

from matchbox.client.helpers.delete import delete_resolution
from matchbox.client.helpers.selector import clean

__all__ = (
    # Deletion
    "delete_resolution",
    # Selectors
    "clean",
)
