from io import BytesIO
from os import getenv

import httpx

from matchbox.common.dtos import QueryParams
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
    qp = QueryParams(
        source_address=source_address,
        resolution_id=resolution_id,
        threshold=threshold,
        limit=limit,
    ).model_dump_json()

    res = httpx.get(url("/query/"), json=qp)
    return BytesIO(res.content)
