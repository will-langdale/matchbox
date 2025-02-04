from io import BytesIO
from os import getenv

import httpx
from pyarrow.parquet import read_table

from matchbox.common.arrow import SCHEMA_MB_IDS
from matchbox.common.dtos import BackendRetrievableType, NotFoundError
from matchbox.common.exceptions import (
    MatchboxClientFileError,
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
    MatchboxUnhandledServerResponse,
    MatchboxUnparsedClientRequest,
)
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import hash_to_base64
from matchbox.common.sources import SourceAddress


def url(path: str) -> str:
    """Return path prefixed by API root, determined from environment"""
    api_root = getenv("API__ROOT")
    if api_root is None:
        raise RuntimeError("API__ROOT needs to be defined in the environment")

    return api_root + path


def url_params(params: dict[str, str | int | float | bytes]) -> dict[str, str]:
    """Prepares a dictionary of parameters to be encoded in a URL"""

    def process_val(v):
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        elif isinstance(v, float):
            return str(v)
        elif isinstance(v, bytes):
            return hash_to_base64(v)

        raise ValueError(f"It was not possible to parse {v} as an URL parameter")

    non_null = {k: v for k, v in params.items() if v}
    return {k: process_val(v) for k, v in non_null.items()}


def handle_http_code(res: httpx.Response) -> httpx.Response:
    if res.status_code == 200:
        return res

    if res.status_code == 404:
        error = NotFoundError.model_validate(res.json())
        if error.entity == BackendRetrievableType.SOURCE:
            raise MatchboxSourceNotFoundError(error.details)
        if error.entity == BackendRetrievableType.RESOLUTION:
            raise MatchboxResolutionNotFoundError(error.details)
        else:
            raise RuntimeError(f"Unexpected 404 error: {error.details}")

    if res.status_code == 422:
        raise MatchboxUnparsedClientRequest(res.content)

    raise MatchboxUnhandledServerResponse(res.content)


def get_resolution_graph() -> ResolutionGraph:
    res = handle_http_code(httpx.get(url("/report/resolutions")))
    return ResolutionGraph.model_validate(res.json())


def query(
    source_address: SourceAddress,
    resolution_id: int | None = None,
    threshold: int | None = None,
    limit: int | None = None,
) -> BytesIO:
    res = handle_http_code(
        httpx.get(
            url("/query"),
            params=url_params(
                {
                    "full_name": source_address.full_name,
                    # Converted to b64 by `url_params()`
                    "warehouse_hash_b64": source_address.warehouse_hash,
                    "resolution_id": resolution_id,
                    "threshold": threshold,
                    "limit": limit,
                }
            ),
        )
    )

    buffer = BytesIO(res.content)
    table = read_table(buffer)

    if not table.schema.equals(SCHEMA_MB_IDS):
        raise MatchboxClientFileError(
            message=(
                f"Schema mismatch. Expected:\n{SCHEMA_MB_IDS}\nGot:\n{table.schema}"
            )
        )

    return table
