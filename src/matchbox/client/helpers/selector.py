import itertools
from typing import Literal
from warnings import warn

import pyarrow as pa
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine

from matchbox.common.db import Match
from matchbox.common.sources import SourceNameAddress
from matchbox.server import MatchboxDBAdapter, inject_backend


class Selector(BaseModel):
    source_name_address: SourceNameAddress
    fields: list[str]
    engine: Engine

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    @inject_backend
    def verify(
        cls,
        backend: MatchboxDBAdapter,
        full_name: str,
        fields: list[str],
        engine: Engine,
    ):
        source_address = SourceNameAddress.compose(engine, full_name)
        source = backend.get_source(source_address)

        warehouse_cols = set(source.to_table().columns.keys())
        selected_cols = set(fields)
        if not selected_cols <= warehouse_cols:
            raise ValueError(
                f"{selected_cols - warehouse_cols} not found in {source_address}"
            )

        indexed_cols = set(col.literal for col in source.columns)
        if not selected_cols <= indexed_cols:
            warn(
                "You are selecting columns that are not indexed in Matchbox",
                stacklevel=2,
            )

        return Selector(source_address, fields, engine)


def select(selection: dict[str, list[str]], engine: Engine) -> list[Selector]:
    """
    Builds and verifies a list of selectors from one engine.

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
    resolution_name: str,
    *selectors: list[Selector],
    return_type: Literal["pandas", "arrow"] = "pandas",
    threshold: float | dict[str, float] | None = None,
    limit: int | None = None,
) -> DataFrame | pa.Table:
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

    selectors = itertools.chain(*selectors)
    resolution_id = backend.get_resolution_id(resolution_name)

    # Divide the limit among selectors
    n_selectors = len(selectors)
    sub_limit_base = limit // n_selectors
    sub_limit_remainder = limit % n_selectors
    sub_limits = [sub_limit_base + 1] * sub_limit_remainder + [sub_limit_base] * (
        n_selectors - sub_limit_remainder
    )

    tables = list[pa.Table]
    for selector, sub_limit in zip(selectors, sub_limits, strict=True):
        # Get ids from matchbox
        mb_ids = backend.query(
            source_address=selector.source_name_address,
            resolution_id=resolution_id,
            threshold=threshold,
            limit=sub_limit,
        )

        # Get source data
        source = backend.get_source(selector.source_name_address)
        source.engine = selector.engine

        raw_data = source.to_arrow(
            fields=set([source.db_pk] + selector.fields),
            pks=mb_ids["source_pk"].to_pylist(),
        )

        # Join and select columns
        joined_table = raw_data.join(
            right_table=mb_ids,
            keys=source.format_column(source.db_pk),
            right_keys="source_pk",
            join_type="inner",
        )

        keep_cols = ["id"] + [source.format_column(f) for f in selector.fields]
        match_cols = [col for col in joined_table.column_names if col in keep_cols]

        tables.append(joined_table.select(match_cols))

    # Combine results
    result = pa.concat_tables(tables, promote_options="default")

    # Return in requested format
    if return_type == "arrow":
        return result
    elif return_type == "pandas":
        return result.to_pandas(
            use_threads=True,
            split_blocks=True,
            self_destruct=True,
            types_mapper=pa.ArrowDtype,
        )
    else:
        raise ValueError(f"return_type of {return_type} not valid")


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
