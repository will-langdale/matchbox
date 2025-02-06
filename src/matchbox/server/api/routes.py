from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator

import pyarrow as pa
from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendCountableType,
    BackendRetrievableType,
    BackendUploadType,
    CountResult,
    HealthCheck,
    ModelResultsType,
    NotFoundError,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import base64_to_hash
from matchbox.common.sources import Source, SourceAddress
from matchbox.server.api.arrow import s3_to_recordbatch, table_to_s3
from matchbox.server.api.cache import MetadataStore
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


metadata_store = MetadataStore(expiry_minutes=30)

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


@app.post("/sources")
async def add_source(source: Source):
    """Add a source to the backend."""
    upload_id = metadata_store.cache_source(metadata=source)
    return UploadStatus(
        id=upload_id, status="awaiting_upload", entity=BackendUploadType.INDEX
    )


@app.post("/upload/{upload_id}")
async def upload_file(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    upload_id: str,
    file: UploadFile,
):
    """Upload file and process based on metadata type.

    Raises HTTP 400 if:

    * Upload ID not found or expired.
    * Uploaded data doesn't match the metadata schema
    * Uploaded metadata is of a type not handled by this endpoint
    """
    source_cache = metadata_store.get(cache_id=upload_id)
    if not source_cache:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                id=upload_id,
                status="failed",
                details="Upload ID not found or expired.",
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )

    # Upload to S3
    client = backend.settings.datastore.get_client()
    bucket = backend.settings.datastore.cache_bucket_name
    key = f"{upload_id}.parquet"
    try:
        upload_id = await table_to_s3(
            client=client,
            bucket=bucket,
            key=key,
            file=file,
            expected_schema=source_cache.upload_schema.value,
        )
    except MatchboxServerFileError as e:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                id=upload_id,
                status="failed",
                details=f"{str(e)}",
                entity=source_cache.upload_type,
            ).model_dump(),
        ) from e

    # Read from S3
    data_hashes = pa.Table.from_batches(
        [
            batch
            async for batch in s3_to_recordbatch(client=client, bucket=bucket, key=key)
        ]
    )

    # Index
    backend.index(source=source_cache.metadata, data_hashes=data_hashes)

    # Clean up
    metadata_store.remove(upload_id)

    return UploadStatus(
        id=upload_id, status="complete", entity=source_cache.upload_type
    )


@app.get("/sources")
async def list_sources():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/sources/{address}")
async def get_source(address: str):
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
    threshold: float | None = None,
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
                details=f"{str(e)}", entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except MatchboxSourceNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=f"{str(e)}", entity=BackendRetrievableType.SOURCE
            ).model_dump(),
        ) from e

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@app.get("/match")
async def match():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/validate/hash")
async def validate_hashes():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/report/resolutions")
async def get_resolutions(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
) -> ResolutionGraph:
    return backend.get_resolution_graph()
