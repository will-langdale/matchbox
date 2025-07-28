"""Functions to delete resolutions from the Matchbox server."""

from matchbox.client import _handler
from matchbox.common.graph import ResolutionName


def delete_resolution(name: ResolutionName, certain: bool = False) -> None:
    """Deletes a resolution from Matchbox.

    Will delete:

    * The resolution itself
    * All descendants of the resolution
    * All endorsements of clusters made by those resolutions, either
        probabilities for models, or keys for sources

    Will not delete:

    * The clusters themselves

    Args:
        name: The name of the source to delete.
        certain: Must be true to delete the source. Default is False.
    """
    _handler.delete_resolution(name=name, certain=certain)
