import time
from collections.abc import Iterable
from io import BytesIO
from os import getenv

import httpx
from pyarrow import Table
from pyarrow.parquet import read_table

from matchbox.common.arrow import SCHEMA_MB_IDS, table_to_buffer
from matchbox.common.dtos import BackendRetrievableType, NotFoundError, UploadStatus
from matchbox.common.exceptions import (
    MatchboxClientFileError,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
    MatchboxUnhandledServerResponse,
    MatchboxUnparsedClientRequest,
)
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import hash_to_base64
from matchbox.common.sources import Match, Source, SourceAddress


def url(path: str) -> str:
    """Return path prefixed by API root, determined from environment."""
    api_root = getenv("MB__API_ROOT")
    if api_root is None:
        raise RuntimeError("MB__API_ROOT needs to be defined in the environment")

    return api_root + path


URLEncodeHandledType = str | int | float | bytes


def encode_param_value(
    v: URLEncodeHandledType | Iterable[URLEncodeHandledType],
) -> str | list[str]:
    if isinstance(v, str):
        return v
    elif isinstance(v, (int, float)):
        return str(v)
    elif isinstance(v, bytes):
        return hash_to_base64(v)
    # Needs to be at the end, so we don't apply it to e.g. strings
    if isinstance(v, Iterable):
        return [encode_param_value(item) for item in v]
    raise ValueError(f"It was not possible to parse {v} as an URL parameter")


def url_params(
    params: dict[str, URLEncodeHandledType | Iterable[URLEncodeHandledType]],
) -> dict[str, str | list[str]]:
    """Prepares a dictionary of parameters to be encoded in a URL"""

    non_null = {k: v for k, v in params.items() if v}
    return {k: encode_param_value(v) for k, v in non_null.items()}


def handle_http_code(res: httpx.Response) -> httpx.Response:
    """Handle HTTP status codes and raise appropriate exceptions."""
    if res.status_code == 200:
        return res

    if res.status_code == 400:
        if UploadStatus.model_validate_json(res.content, strict=False):
            error = UploadStatus.model_validate(res.json())
            raise MatchboxServerFileError(error.details)
        else:
            raise RuntimeError(f"Unexpected 400 error: {res.content}")

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


# Retrieval


def query(
    source_address: SourceAddress,
    resolution_name: str | None = None,
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
                    "resolution_name": resolution_name,
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


def match(
    targets: list[SourceAddress],
    source: SourceAddress,
    source_pk: str,
    resolution_name: str,
    threshold: int | None = None,
) -> Match:
    target_full_names = [t.full_name for t in targets]
    target_warehouse_hashes = [t.warehouse_hash for t in targets]

    res = handle_http_code(
        httpx.get(
            url("/match"),
            params=url_params(
                {
                    "target_full_names": target_full_names,
                    # Converted to b64 by `url_params()`
                    "target_warehouse_hashes_b64": target_warehouse_hashes,
                    "source_full_name": source.full_name,
                    # Converted to b64 by `url_params()`
                    "source_warehouse_hash_b64": source.warehouse_hash,
                    "source_pk": source_pk,
                    "resolution_name": resolution_name,
                    "threshold": threshold,
                }
            ),
        )
    )

    return [Match.model_validate(m) for m in res.json()]


# Data management


def index(source: Source, data_hashes: Table) -> UploadStatus:
    """Index a Source in Matchbox."""
    buffer = table_to_buffer(table=data_hashes)

    # Upload metadata
    metadata_res = handle_http_code(
        httpx.post(url("/sources"), json=source.model_dump())
    )
    upload = UploadStatus.model_validate(metadata_res.json())

    # Upload data
    upload_res = handle_http_code(
        httpx.post(
            url(f"/upload/{upload.id}"),
            files={
                "file": (f"{upload.id}.parquet", buffer, "application/octet-stream")
            },
        )
    )

    # Poll until complete with retry/timeout configuration
    status = UploadStatus.model_validate(upload_res.json())
    while status.status not in ["complete", "failed"]:
        status_res = handle_http_code(httpx.get(url(f"/upload/{upload.id}/status")))
        status = UploadStatus.model_validate(status_res.json())

        if status.status == "failed":
            raise MatchboxServerFileError(status.details)

        time.sleep(2)

    return status


def get_source(address: SourceAddress) -> Source:
    warehouse_hash_b64 = hash_to_base64(address.warehouse_hash)
    res = handle_http_code(
        httpx.get(url(f"/sources/{warehouse_hash_b64}/{address.full_name}"))
    )
    return Source.model_validate(res.json())


def get_resolution_graph() -> ResolutionGraph:
    """Get the resolution graph from Matchbox."""
    res = handle_http_code(httpx.get(url("/report/resolutions")))
    return ResolutionGraph.model_validate(res.json())


# Model management
