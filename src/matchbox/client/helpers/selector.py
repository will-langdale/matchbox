"""Functions to select and retrieve data from the Matchbox server."""

import itertools
from typing import Iterator, Literal, get_args

import polars as pl
import pyarrow as pa
from pandas import ArrowDtype
from pyarrow import Table as ArrowTable
from pyarrow import compute as pc
from pydantic import BaseModel
from sqlalchemy import (
    Engine,
)

from matchbox.client import _handler
from matchbox.client.warehouse import SourceReader, engine_fallback
from matchbox.common.db import (
    QueryReturnType,
    ReturnTypeStr,
)
from matchbox.common.graph import DEFAULT_RESOLUTION
from matchbox.common.sources import Match, SourceAddress


class Selector(BaseModel):
    """A full source name, an engine and optionally a subset of fields to select."""

    full_name: str
    engine: Engine
    fields: list[str] | None = None


def select(
    *selection: str | dict[str, str],
    engine: Engine | None = None,
) -> list[Selector]:
    """From one engine, refer to a selection of source names and fields.

    Args:
        selection: Full source names and optionally a subset of columns to select
        engine: The engine to connect to the data warehouse hosting the source.
            If not provided, will use a connection string from the
            `MB__CLIENT__DEFAULT_WAREHOUSE` environment variable.

    Returns:
        A list of SourceReader objects

    Examples:
        ```python
        select("companies_house", "hmrc_exporters", engine=engine)
        ```

        ```python
        select({"companies_house": ["crn"], "hmrc_exporters": ["name"]}, engine=engine)
        ```
    """
    engine = engine_fallback(engine)

    selectors = []
    for s in selection:
        if isinstance(s, str):
            selectors.append(Selector(full_name=s, engine=engine))

        elif isinstance(selection, dict):
            for full_name, fields in s.items():
                selectors.append(
                    Selector(full_name=full_name, fields=fields, engine=engine)
                )
        else:
            raise ValueError("Selection specified in incorrect format")

    return selectors


def _process_query_result(
    data: ArrowTable, reader: SourceReader, mb_ids: ArrowTable
) -> ArrowTable:
    """Process query results by joining with matchbox IDs and filtering fields.

    Args:
        data: The raw data from the source
        reader: The `SourceReader` used to access the source
        mb_ids: The matchbox IDs

    Returns:
        The processed table with joined matchbox IDs and filtered fields
    """
    # Join data with matchbox IDs
    joined_table = data.join(
        right_table=mb_ids,
        keys=reader.format_column(reader.db_pk),
        right_keys="source_pk",
        join_type="inner",
    )

    return joined_table


def _query_batched(
    readers: list[SourceReader],
    sub_limits: list[int | None],
    resolution_name: str | None,
    threshold: int | None,
    return_type: ReturnTypeStr,
    batch_size: int | None,
) -> Iterator[QueryReturnType]:
    """Helper function that implements batched query processing.

    Returns an iterator yielding batches of results.
    """
    # Create iterators for each reader
    reader_iters = []

    for reader, sub_limit in zip(readers, sub_limits, strict=True):
        # Get ids from matchbox
        mb_ids = _handler.query(
            source_address=reader.address,
            resolution_name=resolution_name,
            threshold=threshold,
            limit=sub_limit,
        )

        # Get batched data
        raw_batches = reader.data(
            pks=mb_ids["source_pk"].to_pylist(),
            iter_batches=True,
            batch_size=batch_size,
        )

        # Process and transform each batch
        def process_batches(batches, reader, mb_ids):
            for batch in batches:
                yield _process_query_result(batch, reader, mb_ids)

        reader_iters.append(process_batches(raw_batches, reader, mb_ids))

    # Chain iterators if multiple readers
    if len(reader_iters) == 1:
        batches_iter = reader_iters[0]
    else:
        # Interleave batches from different readers
        batches_iter: list[ArrowTable] = itertools.chain(*reader_iters)

    # Convert each batch to the requested return type
    for batch in batches_iter:
        match return_type:
            case "pandas":
                yield batch.to_pandas(
                    use_threads=True,
                    split_blocks=True,
                    self_destruct=True,
                    types_mapper=ArrowDtype,
                )
            case "polars":
                yield pl.from_arrow(batch)
            case "arrow":
                yield batch


def query(
    *selectors: list[Selector],
    resolution_name: str | None = None,
    combine_type: Literal["concat", "explode", "set_agg"] = "concat",
    return_type: ReturnTypeStr = "pandas",
    threshold: int | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    return_batches: bool = False,
    only_indexed: bool = False,
) -> QueryReturnType | Iterator[QueryReturnType]:
    """Runs queries against the selected backend.

    Args:
        selectors: Each item is the output of `select()`.
            Use one `select()` call per engine to use.
        resolution_name (optional): The name of the resolution point to query
            If not set, it will look for a default resolution
        combine_type: How to combine the data from different sources.
            * If `concat`, concatenate all sources queried without any merging.
                Multiple rows per ID, with null values where data isn't available
            * If `explode`, outer join on Matchbox ID. Multiple rows per ID,
                with one for every unique combination of data requested
                across all sources
            * If `set_agg`, join on Matchbox ID, group on Matchbox ID, then
                aggregate to nested lists of unique values. One row per ID,
                but all requested data is in nested arrays
        return_type: The form to return data in, one of "pandas" or "arrow"
            Defaults to pandas for ease of use
        threshold (optional): The threshold to use for creating clusters
            If None, uses the resolutions' default threshold
            If an integer, uses that threshold for the specified resolution, and the
            resolution's cached thresholds for its ancestors
        limit (optional): The number to use in a limit clause. Useful for testing
        batch_size (optional): The size of each batch when fetching data from the
            warehouse, which helps reduce memory usage and load on the database.
            Default is None.
        return_batches (optional): If True, returns an iterator of batches instead of a
            single combined result, which is useful for processing large datasets with
            limited memory. Default is False.
        only_indexed: Whether you intend to select indexed columns only. Will raise an
            exception if True and non-indexed columns are selected. Defaults to False.
            Non-indexed columns should only be selected if you're querying data for
            a purpose other than matching.

    Returns:
        If return_batches is False:
            Data in the requested return type (DataFrame or ArrowTable)
        If return_batches is True:
            An iterator yielding batches in the requested return type

    Examples:
        ```python
        query(
            select({"companies_house": ["crn", "name"]}, engine=engine),
        )
        ```

        ```python
        query(
            select("companies_house", engine=engine1),
            select("datahub_companies", engine=engine2),
            resolution_name="last_linker",
        )
        ```

        ```python
        # Process large results in batches of 5000 rows
        for batch in query(
            select("companies_house", engine=engine),
            batch_size=5000,
            return_batches=True,
        ):
            batch.head()
        ```
    """
    # Validate arguments
    if combine_type not in ("concat", "explode", "set_agg"):
        raise ValueError(f"combine_type of {combine_type} not valid")

    if return_type not in get_args(ReturnTypeStr):
        raise ValueError(f"return_type of {return_type} not valid")

    if not selectors:
        raise ValueError("At least one source must be selected")

    selectors: list[Selector] = list(itertools.chain(*selectors))
    configs = [
        _handler.get_source_config(
            SourceAddress.compose(full_name=s.full_name, engine=s.engine)
        )
        for s in selectors
    ]

    readers: list[SourceReader] = []
    for selector, config in zip(selectors, configs, strict=True):
        selected_fields = set(selector.fields)
        indexed_fields = set(col.name for col in config.columns)

        if (not selected_fields <= indexed_fields) and only_indexed:
            raise ValueError("Cannot select non-indexed columns with only_indexed=True")

        readers.append(
            SourceReader(
                engine=selector.engine,
                full_name=selector.full_name,
                db_pk=config.db_pk,
                fields=selector.fields,
            )
        )

    if not resolution_name:
        resolution_name = DEFAULT_RESOLUTION

    # Divide the limit among readers
    if limit:
        n_readers = len(readers)
        sub_limit_base = limit // n_readers
        sub_limit_remainder = limit % n_readers
        sub_limits = [sub_limit_base + 1] * sub_limit_remainder + [sub_limit_base] * (
            n_readers - sub_limit_remainder
        )
    else:
        sub_limits = [None] * len(readers)

    if return_batches:
        # Return an iterator of batches
        return _query_batched(
            readers=readers,
            sub_limits=sub_limits,
            resolution_name=resolution_name,
            threshold=threshold,
            return_type=return_type,
            batch_size=batch_size,
        )
    else:
        # Process all data and return a single result
        tables: list[ArrowTable] = []
        for reader, sub_limit in zip(readers, sub_limits, strict=True):
            # Get ids from matchbox
            mb_ids = _handler.query(
                source_address=reader.address,
                resolution_name=resolution_name,
                threshold=threshold,
                limit=sub_limit,
            )

            raw_data = reader.data(
                pks=mb_ids["source_pk"].to_pylist(),
                batch_size=batch_size,
            )

            processed_table = _process_query_result(raw_data, reader, mb_ids)
            tables.append(processed_table)

        # Combine results based on combine_type
        if combine_type == "concat":
            result = pa.concat_tables(tables, promote_options="default")
        else:
            result = tables[0]
            for table in tables[1:]:
                result = result.join(table, keys=["id"], join_type="full outer")

            if combine_type == "set_agg":
                # Aggregate into lists
                aggregate_rule = [
                    (col, "distinct", pc.CountOptions(mode="only_valid"))
                    for col in result.column_names
                    if col != "id"
                ]
                result = result.group_by("id").aggregate(aggregate_rule)
                # Recover original column names
                rename_rule = {f"{col}_distinct": col for col, _, _ in aggregate_rule}
                result = result.rename_columns(rename_rule)

        # Return in requested format
        match return_type:
            case "pandas":
                return result.to_pandas(
                    use_threads=True,
                    split_blocks=True,
                    self_destruct=True,
                    types_mapper=ArrowDtype,
                )
            case "polars":
                return pl.from_arrow(result)
            case "arrow":
                return result


def match(
    *targets: list[SourceReader],
    source: list[SourceReader],
    source_pk: str,
    resolution_name: str = DEFAULT_RESOLUTION,
    threshold: int | None = None,
) -> list[Match]:
    """Matches IDs against the selected backend.

    Args:
        targets: Each target is the output of `select()`.
            This allows matching against sources coming from different engines
        source: The output of using `select()` on a single source.
        source_pk: The primary key value to match from the source.
        resolution_name (optional): The resolution name to use for filtering results.
            If not set, it will look for a default resolution.
        threshold (optional): The threshold to use for creating clusters.
            If None, uses the resolutions' default threshold
            If an integer, uses that threshold for the specified resolution, and the
            resolution's cached thresholds for its ancestors

    Examples:
        ```python
        mb.match(
            select("datahub_companies", engine=engine),
            source=select("companies_house", engine=engine),
            source_pk="8534735",
            resolution_name="last_linker",
        )
        ```
    """
    if len(source) > 1:
        raise ValueError("Only one source can be matched at one time")
    source = source[0].address

    targets: list[SourceReader] = list(itertools.chain(*targets))
    targets = [t.address for t in targets]

    return _handler.match(
        targets=targets,
        source=source,
        source_pk=source_pk,
        resolution_name=resolution_name,
        threshold=threshold,
    )
