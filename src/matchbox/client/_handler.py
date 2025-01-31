from io import BytesIO
from os import getenv

import httpx
from pyarrow.parquet import read_table

from matchbox.common.arrow import SCHEMA_MB_IDS
from matchbox.common.exceptions import (
    MatchboxClientFileError,
    MatchboxServerResolutionError,
)
from matchbox.common.graph import ResolutionGraph
from matchbox.common.sources import SourceAddress


def url(path: str) -> str:
    """Return path prefixed by API root, determined from environment"""
    api_root = getenv("API__ROOT")
    if api_root is None:
        raise RuntimeError("API__ROOT needs to be defined in the environment")

    return api_root + path


def get_resolution_graph() -> ResolutionGraph:
    res = httpx.get(url("/report/resolutions")).json()
    return ResolutionGraph.model_validate(res)


def query(
    source_address: SourceAddress,
    resolution_id: int | None = None,
    threshold: int | None = None,
    limit: int | None = None,
) -> BytesIO:
    res = httpx.get(
        url("/query"),
        params={
            "full_name": source_address.full_name,
            "warehouse_hash": source_address.warehouse_hash,
            "resolution_id": resolution_id,
            "threshold": threshold,
            "limit": limit,
        },
    )

    if res.status_code == 404:
        raise MatchboxServerResolutionError(res.json().detail)

    buffer = BytesIO(res.content)
    table = read_table(buffer)

    if not table.schema.equals(SCHEMA_MB_IDS):
        raise MatchboxClientFileError(
            message=(
                f"Schema mismatch. Expected:\n{SCHEMA_MB_IDS}\nGot:\n{table.schema}"
            )
        )

    return table
