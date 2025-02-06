from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendCountableType,
    BackendRetrievableType,
    CountResult,
    HealthCheck,
    ModelResultsType,
    NotFoundError,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import base64_to_hash
from matchbox.common.sources import Match, SourceAddress
from matchbox.server.base import BackendManager, MatchboxDBAdapter

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


class ParquetResponse(Response):
    media_type = "application/octet-stream"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    get_backend()
    yield


app = FastAPI(
    title="matchbox API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Overwrite the default JSON schema for an `HTTPException`"""
    return JSONResponse(content=exc.detail, status_code=exc.status_code)


def get_backend() -> MatchboxDBAdapter:
    return BackendManager.get_backend()


async def table_to_s3(
    client: S3Client, bucket: str, file: UploadFile, expected_schema: pa.Schema
) -> str:
    """Upload a PyArrow Table to S3 and return the key.

    Args:
        client: The S3 client to use.
        bucket: The S3 bucket to upload to.
        file: The file to upload.
        expected_schema: The schema that the file should match.

    Raises:
        MatchboxServerFileError: If the file is not a valid Parquet file or the schema
            does not match the expected schema.

    Returns:
        The key of the uploaded file.
    """
    upload_id = str(uuid4())
    key = f"{upload_id}.parquet"

    try:
        table = pq.read_table(file.file)

        if not table.schema.equals(expected_schema):
            raise MatchboxServerFileError(
                message=(
                    "Schema mismatch. "
                    f"Expected:\n{expected_schema}\nGot:\n{table.schema}"
                )
            )

        await file.seek(0)

        client.put_object(Bucket=bucket, Key=key, Body=file.file)

    except Exception as e:
        if isinstance(e, MatchboxServerFileError):
            raise
        raise MatchboxServerFileError(message=f"Invalid Parquet file: {str(e)}") from e

    return upload_id


async def s3_to_recordbatch(
    client: S3Client, bucket: str, key: str, batch_size: int = 1000
) -> AsyncGenerator[pa.RecordBatch, None]:
    """Download a PyArrow Table from S3 and stream it as RecordBatches."""
    response = client.get_object(Bucket=bucket, Key=key)
    buffer = pa.BufferReader(response["Body"].read())

    parquet_file = pq.ParquetFile(buffer)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch


@app.get("/health")
async def healthcheck() -> HealthCheck:
    """Perform a health check and return the status."""
    return HealthCheck(status="OK")


@app.get("/testing/count")
async def count_backend_items(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    entity: BackendCountableType | None = None,
) -> CountResult:
    def get_count(e: BackendCountableType) -> int:
        return getattr(backend, str(e)).count()

    if entity is not None:
        return CountResult(entities={str(entity): get_count(entity)})
    else:
        res = {str(e): get_count(e) for e in BackendCountableType}
        return CountResult(entities=res)


@app.post("/testing/clear")
async def clear_backend():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/sources")
async def list_sources():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/sources/{address}")
async def get_source(address: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/sources/{hash}")
async def add_source(hash: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models")
async def list_models():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/resolution/{name}")
async def get_resolution(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}")
async def add_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.delete("/models/{name}")
async def delete_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/results")
async def get_results(name: str, result_type: ModelResultsType | None):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/results")
async def set_results(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/truth")
async def get_truth(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/truth")
async def set_truth(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/ancestors")
async def get_ancestors(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/ancestors_cache")
async def get_ancestors_cache(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/ancestors_cache")
async def set_ancestors_cache(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get(
    "/query",
    response_class=ParquetResponse,
    responses={404: NotFoundError.example_response_body()},
)
async def query(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    full_name: str,
    warehouse_hash_b64: str,
    resolution_id: int | None = None,
    threshold: int | None = None,
    limit: int | None = None,
):
    warehouse_hash = base64_to_hash(warehouse_hash_b64)
    source_address = SourceAddress(full_name=full_name, warehouse_hash=warehouse_hash)
    try:
        res = backend.query(
            source_address=source_address,
            resolution_id=resolution_id,
            threshold=threshold,
            limit=limit,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except MatchboxSourceNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.SOURCE
            ).model_dump(),
        ) from e

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@app.get(
    "/match",
    responses={404: NotFoundError.example_response_body()},
)
async def match(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    target_full_names: Annotated[list[str], Query()],
    target_warehouse_hashes_b64: Annotated[list[str], Query()],
    source_full_name: str,
    source_warehouse_hash_b64: str,
    source_pk: str,
    resolution_name: str,
    threshold: int | None = None,
) -> list[Match]:
    targets = [
        SourceAddress(full_name=n, warehouse_hash=w)
        for n, w in zip(target_full_names, target_warehouse_hashes_b64, strict=True)
    ]
    source_warehouse_hash = base64_to_hash(source_warehouse_hash_b64)
    source = SourceAddress(
        full_name=source_full_name, warehouse_hash=source_warehouse_hash
    )
    try:
        res = backend.match(
            source_pk=source_pk,
            source=source,
            targets=targets,
            resolution_name=resolution_name,
            threshold=threshold,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except MatchboxSourceNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.SOURCE
            ).model_dump(),
        ) from e

    return res


@app.get("/validate/hash")
async def validate_hashes():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/report/resolutions")
async def get_resolutions(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
) -> ResolutionGraph:
    return backend.get_resolution_graph()
