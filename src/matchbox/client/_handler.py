import io
from os import getenv

import httpx
import pyarrow.parquet as pq
from pyarrow import Table

from matchbox.common.dtos import SourceStatus
from matchbox.common.graph import ResolutionGraph
from matchbox.common.sources import Source


def url(path: str) -> str:
    """Return path prefixed by API root, determined from environment"""
    api_root = getenv("MB__API_ROOT")
    if api_root is None:
        raise RuntimeError("MB__API_ROOT needs to be defined in the environment")

    return api_root + path


def get_resolution_graph() -> str:
    """Return the resolution subgraph from Matchbox."""
    res = httpx.get(url("/report/resolutions"))
    res.raise_for_status()
    return ResolutionGraph.model_validate(res.json())


def index(source: Source, data_hashes: Table) -> SourceStatus:
    """Index a Source in Matchbox."""
    # Write Table to a buffer
    buffer = io.BytesIO()
    pq.write_table(data_hashes, buffer)
    buffer.seek(0)

    # Post
    res = httpx.post(
        url("/index"),
        data={
            "source": source.model_dump_json(),
        },
        files={
            "data": ("data.parquet", buffer, "application/x-parquet"),
        },
    )
    res.raise_for_status()

    return SourceStatus(**res)
