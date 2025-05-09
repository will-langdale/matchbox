"""Model API routes for the Matchbox server."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
    status,
)

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendRetrievableType,
    ModelAncestor,
    ModelMetadata,
    ModelOperationStatus,
    ModelOperationType,
    ModelResolutionName,
    NotFoundError,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
)
from matchbox.server.api.dependencies import (
    BackendDependency,
    MetadataStoreDependency,
    ParquetResponse,
    validate_api_key,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.post(
    "",
    responses={
        500: {
            "model": ModelOperationStatus,
            **ModelOperationStatus.status_500_examples(),
        },
    },
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(validate_api_key)],
)
async def insert_model(
    backend: BackendDependency, model: ModelMetadata
) -> ModelOperationStatus:
    """Insert a model into the backend."""
    try:
        backend.insert_model(model)
        return ModelOperationStatus(
            success=True,
            name=model.name,
            operation=ModelOperationType.INSERT,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ModelOperationStatus(
                success=False,
                name=model.name,
                operation=ModelOperationType.INSERT,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{name}",
    responses={404: {"model": NotFoundError}},
)
async def get_model(
    backend: BackendDependency, name: ModelResolutionName
) -> ModelMetadata:
    """Get a model from the backend."""
    try:
        return backend.get_model(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e


@router.post(
    "/{name}/results",
    responses={404: {"model": NotFoundError}},
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(validate_api_key)],
)
async def set_results(
    backend: BackendDependency,
    metadata_store: MetadataStoreDependency,
    name: ModelResolutionName,
) -> UploadStatus:
    """Create an upload task for model results."""
    try:
        metadata = backend.get_model(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e

    upload_id = metadata_store.cache_model(metadata=metadata)
    return metadata_store.get(cache_id=upload_id).status


@router.get(
    "/{name}/results",
    responses={404: {"model": NotFoundError}},
)
async def get_results(
    backend: BackendDependency, name: ModelResolutionName
) -> ParquetResponse:
    """Download results for a model as a parquet file."""
    try:
        res = backend.get_model_results(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@router.patch(
    "/{name}/truth",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ModelOperationStatus,
            **ModelOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(validate_api_key)],
)
async def set_truth(
    backend: BackendDependency,
    name: ModelResolutionName,
    truth: Annotated[int, Body(ge=0, le=100)],
) -> ModelOperationStatus:
    """Set truth data for a model."""
    try:
        backend.set_model_truth(name=name, truth=truth)
        return ModelOperationStatus(
            success=True,
            name=name,
            operation=ModelOperationType.UPDATE_TRUTH,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ModelOperationStatus(
                success=False,
                name=name,
                operation=ModelOperationType.UPDATE_TRUTH,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{name}/truth",
    responses={404: {"model": NotFoundError}},
)
async def get_truth(backend: BackendDependency, name: ModelResolutionName) -> float:
    """Get truth data for a model."""
    try:
        return backend.get_model_truth(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e


@router.get(
    "/{name}/ancestors",
    responses={404: {"model": NotFoundError}},
)
async def get_ancestors(
    backend: BackendDependency, name: ModelResolutionName
) -> list[ModelAncestor]:
    """Get the ancestors for a model."""
    try:
        return backend.get_model_ancestors(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e


@router.patch(
    "/{name}/ancestors_cache",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ModelOperationStatus,
            **ModelOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(validate_api_key)],
)
async def set_ancestors_cache(
    backend: BackendDependency,
    name: ModelResolutionName,
    ancestors: list[ModelAncestor],
):
    """Update the cached ancestors for a model."""
    try:
        backend.set_model_ancestors_cache(name=name, ancestors_cache=ancestors)
        return ModelOperationStatus(
            success=True,
            name=name,
            operation=ModelOperationType.UPDATE_ANCESTOR_CACHE,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ModelOperationStatus(
                success=False,
                name=name,
                operation=ModelOperationType.UPDATE_ANCESTOR_CACHE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{name}/ancestors_cache",
    responses={404: {"model": NotFoundError}},
)
async def get_ancestors_cache(
    backend: BackendDependency, name: ModelResolutionName
) -> list[ModelAncestor]:
    """Get the cached ancestors for a model."""
    try:
        return backend.get_model_ancestors_cache(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e


@router.delete(
    "/{name}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ModelOperationStatus,
            **ModelOperationStatus.status_409_examples(),
        },
    },
    dependencies=[Depends(validate_api_key)],
)
async def delete_model(
    backend: BackendDependency,
    name: ModelResolutionName,
    certain: Annotated[
        bool, Query(description="Confirm deletion of the model")
    ] = False,
) -> ModelOperationStatus:
    """Delete a model from the backend."""
    try:
        backend.delete_model(name=name, certain=certain)
        return ModelOperationStatus(
            success=True,
            name=name,
            operation=ModelOperationType.DELETE,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except MatchboxDeletionNotConfirmed as e:
        raise HTTPException(
            status_code=409,
            detail=ModelOperationStatus(
                success=False,
                name=name,
                operation=ModelOperationType.DELETE,
                details=str(e),
            ).model_dump(),
        ) from e
