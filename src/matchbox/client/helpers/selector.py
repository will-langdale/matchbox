from typing import Literal

from pandas import DataFrame
from pyarrow import Table as ArrowTable
from sqlalchemy import Engine

from matchbox.common.db import Match
from matchbox.common.sources import Selector
from matchbox.server import MatchboxDBAdapter, inject_backend


def select(selection: dict[str, list[str]], engine: Engine) -> list[Selector]:
    """
    Builds and verifies a list of selectors.

    Args:
        selection: a dict where full source names are mapped to lists of fields
        engine: the engine to connect to a data warehouse

    Returns:
        A list of Selector objects
    """
    return [
        Selector.verify(engine, full_name, fields) for full_name, fields in selection
    ]


@inject_backend
def query(
    backend: MatchboxDBAdapter,
    selectors: list[Selector],
    return_type: Literal["pandas", "arrow"] = None,
    resolution: str | None = None,
    threshold: float | dict[str, float] | None = None,
    limit: int | None = None,
) -> DataFrame | ArrowTable:
    """Runs queries against the selected backend.

    Args:
        backend: the backend to query
        selector: the tables and fields to query
        return_type: the form to return data in, one of "pandas" or "arrow"
            Defaults to pandas for ease of use
        resolution (optional): the resolution to use for filtering results
        threshold (optional): the threshold to use for creating clusters
            If None, uses the resolutions' default threshold
            If a float, uses that threshold for the specified resolution, and the
            resolution's cached thresholds for its ancestors
            If a dictionary, expects a shape similar to resolution.ancestors, keyed
            by resolution name and valued by the threshold to use for that resolution.
            Will use these threshold values instead of the cached thresholds
        limit (optional): the number to use in a limit clause. Useful for testing

    Returns:
        Data in the requested return type
    """
    # TODO: Logic for joining the results will have to be moved here
    #
    return backend.query(
        selector=selectors,
        resolution=resolution,
        threshold=threshold,
        return_type="pandas" if not return_type else return_type,
        limit=limit,
    )


@inject_backend
def match(
    backend: MatchboxDBAdapter,
    source_pk: str,
    source: str,
    target: str | list[str],
    resolution: str,
    threshold: float | dict[str, float] | None = None,
) -> Match | list[Match]:
    """Matches IDs against the selected backend.

    Args:
        backend: the backend to query
        source_pk: The primary key to match from the source.
        source: The name of the source dataset.
        target: The name of the target dataset(s).
        resolution: the resolution to use for filtering results
        threshold (optional): the threshold to use for creating clusters
            If None, uses the resolutions' default threshold
            If a float, uses that threshold for the specified resolution, and the
            resolution's cached thresholds for its ancestors
            If a dictionary, expects a shape similar to resolution.ancestors, keyed
            by resolution name and valued by the threshold to use for that resolution.
            Will use these threshold values instead of the cached thresholds
    """
    return backend.match(
        source_pk=source_pk,
        source=source,
        target=target,
        resolution=resolution,
        threshold=threshold,
    )
