from io import BytesIO

from pyarrow import Table
from pyarrow.parquet import read_table

from matchbox.client._handler.main import CLIENT, http_retry, url_params
from matchbox.common.arrow import (
    SCHEMA_QUERY,
    SCHEMA_QUERY_WITH_LEAVES,
    check_schema,
)
from matchbox.common.dtos import (
    Match,
    ResolverStepPath,
    SourceStepPath,
)
from matchbox.common.exceptions import MatchboxEmptyServerResponse
from matchbox.common.logging import logger


@http_retry
def query(
    source: SourceStepPath,
    return_leaf_id: bool,
    resolver: ResolverStepPath | None = None,
    limit: int | None = None,
) -> Table:
    """Query a source in Matchbox."""
    log_prefix = f"Query {source}"
    logger.debug(f"Using {resolver}", prefix=log_prefix)

    params = url_params(
        {
            "collection": source.collection,
            "run_id": source.run,
            "source": source.name,
            "resolver": resolver.name if resolver else None,
            "return_leaf_id": return_leaf_id,
            "limit": limit,
        }
    )
    res = CLIENT.get("/query", params=params)

    buffer = BytesIO(res.content)
    table = read_table(buffer)

    logger.debug("Finished", prefix=log_prefix)

    expected_schema = SCHEMA_QUERY_WITH_LEAVES if return_leaf_id else SCHEMA_QUERY

    check_schema(expected_schema, table.schema)

    if table.num_rows == 0:
        raise MatchboxEmptyServerResponse(operation="query")

    return table


@http_retry
def match(
    targets: list[SourceStepPath],
    source: SourceStepPath,
    key: str,
    resolver: ResolverStepPath,
) -> list[Match]:
    """Match a source against a list of targets."""
    log_prefix = f"Query {source}"
    target_names = ", ".join(str(target) for target in targets)
    logger.debug(
        f"{key} to {target_names} using {resolver}",
        prefix=log_prefix,
    )

    params = url_params(
        {
            "collection": resolver.collection,
            "run_id": resolver.run,
            "targets": [t.name for t in targets],
            "source": source.name,
            "key": key,
            "resolver": resolver.name,
        }
    )
    res = CLIENT.get("/match", params=params)

    logger.debug("Finished", prefix=log_prefix)

    matches = [Match.model_validate(m) for m in res.json()]

    if not matches:
        raise MatchboxEmptyServerResponse(operation="match")

    return matches
