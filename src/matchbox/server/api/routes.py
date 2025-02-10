from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator

from dotenv import find_dotenv, load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, UploadFile
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
from matchbox.common.sources import Match, Source, SourceAddress
from matchbox.server.api.arrow import table_to_s3
from matchbox.server.api.cache import MetadataStore, process_upload
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


@app.get("/sources")
async def list_sources():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get(
    "/sources/{warehouse_hash_b64}/{full_name}",
    responses={404: NotFoundError.response_schema()},
)
async def get_source(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    warehouse_hash_b64: str,
    full_name: str,
) -> Source:
    address = SourceAddress(full_name=full_name, warehouse_hash=warehouse_hash_b64)
    try:
        return backend.get_source(address)
    except MatchboxSourceNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.SOURCE
            ).model_dump(),
        ) from e


@app.post("/sources")
async def add_source(source: Source):
    """Add a source to the backend."""
    upload_id = metadata_store.cache_source(metadata=source)
    return metadata_store.get(cache_id=upload_id).status


@app.post(
    "/upload/{upload_id}",
    responses={400: UploadStatus.status_400_response_schema()},
)
async def upload_file(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    background_tasks: BackgroundTasks,
    upload_id: str,
    file: UploadFile,
) -> UploadStatus:
    """Upload and process a file based on metadata type.

    The file is uploaded to S3 and then processed in a background task.
    Status can be checked using the /upload/{upload_id}/status endpoint.

    Raises HTTP 400 if:
    * Upload ID not found or expired (entries expire after 30 minutes of inactivity)
    * Upload is already being processed
    * Uploaded data doesn't match the metadata schema
    """
    # Get and validate cache entry
    source_cache = metadata_store.get(cache_id=upload_id)
    if not source_cache:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                id=upload_id,
                status="failed",
                details=(
                    "Upload ID not found or expired. Entries expire after 30 minutes "
                    "of inactivity, including failed processes."
                ),
            ).model_dump(),
        )

    # Check if already processing
    if source_cache.status.status != "awaiting_upload":
        raise HTTPException(
            status_code=400,
            detail=source_cache.status.model_dump(),
        )

    # Upload to S3
    client = backend.settings.datastore.get_client()
    bucket = backend.settings.datastore.cache_bucket_name
    key = f"{upload_id}.parquet"

    try:
        await table_to_s3(
            client=client,
            bucket=bucket,
            key=key,
            file=file,
            expected_schema=source_cache.upload_type.schema,
        )
    except MatchboxServerFileError as e:
        metadata_store.update_status(upload_id, "failed", details=str(e))
        raise HTTPException(
            status_code=400,
            detail=source_cache.status.model_dump(),
        ) from e

    metadata_store.update_status(upload_id, "queued")

    # Start background processing
    background_tasks.add_task(
        process_upload,
        backend=backend,
        upload_id=upload_id,
        bucket=bucket,
        key=key,
        metadata_store=metadata_store,
    )

    return metadata_store.get(upload_id).status


@app.get(
    "/upload/{upload_id}/status",
    responses={400: UploadStatus.status_400_response_schema()},
)
async def get_upload_status(
    upload_id: str,
) -> UploadStatus:
    """Get the status of an upload process.

    Returns the current status of the upload and processing task.

    Raises HTTP 400 if:
    * Upload ID not found or expired (entries expire after 30 minutes of inactivity)
    """
    source_cache = metadata_store.get(cache_id=upload_id)
    if not source_cache:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                id=upload_id,
                status="failed",
                details=(
                    "Upload ID not found or expired. Entries expire after 30 minutes "
                    "of inactivity, including failed processes."
                ),
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )

    return source_cache.status


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
    responses={404: NotFoundError.response_schema()},
)
async def query(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    full_name: str,
    warehouse_hash_b64: str,
    resolution_name: str | None = None,
    threshold: int | None = None,
    limit: int | None = None,
):
    source_address = SourceAddress(
        full_name=full_name, warehouse_hash=warehouse_hash_b64
    )
    try:
        res = backend.query(
            source_address=source_address,
            resolution_name=resolution_name,
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
    responses={404: NotFoundError.response_schema()},
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
    source = SourceAddress(
        full_name=source_full_name, warehouse_hash=source_warehouse_hash_b64
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
