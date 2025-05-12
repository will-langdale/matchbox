"""SourceConfig API routes for the Matchbox server."""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from matchbox.common.dtos import (
    BackendRetrievableType,
    NotFoundError,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.sources import SourceAddress, SourceConfig
from matchbox.server.api.dependencies import (
    BackendDependency,
    MetadataStoreDependency,
    validate_api_key,
)

router = APIRouter(prefix="/sources", tags=["sources"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(validate_api_key)],
)
async def add_source(
    metadata_store: MetadataStoreDependency, source: SourceConfig
) -> UploadStatus:
    """Create an upload and insert task for indexed source data."""
    upload_id = metadata_store.cache_source(metadata=source)
    return metadata_store.get(cache_id=upload_id).status


@router.get(
    "/{warehouse_hash_b64}/{full_name}",
    responses={404: {"model": NotFoundError}},
)
async def get_source_config(
    backend: BackendDependency,
    warehouse_hash_b64: str,
    full_name: str,
) -> SourceConfig:
    """Get a source from the backend."""
    address = SourceAddress(full_name=full_name, warehouse_hash=warehouse_hash_b64)
    try:
        return backend.get_source_config(address)
    except MatchboxSourceNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.SOURCE
            ).model_dump(),
        ) from e


@router.get(
    "",
    responses={404: {"model": NotFoundError}},
)
async def get_resolution_source_configs(
    backend: BackendDependency,
    resolution_name: str,
) -> list[SourceConfig]:
    """Get all sources in scope for a resolution."""
    try:
        return backend.get_resolution_source_configs(resolution_name=resolution_name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
