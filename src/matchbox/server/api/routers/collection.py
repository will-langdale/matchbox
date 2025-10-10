"""Collection and resolution API routes for the Matchbox server."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    Collection,
    CollectionName,
    CRUDOperation,
    ModelResolutionName,
    NotFoundError,
    Resolution,
    ResolutionName,
    ResolutionPath,
    ResolutionType,
    ResourceOperationStatus,
    Run,
    RunID,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxCollectionNotFoundError,
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionAlreadyExists,
    MatchboxResolutionNotFoundError,
    MatchboxRunNotFoundError,
)
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    UploadTrackerDependency,
    authorisation_dependencies,
)

router = APIRouter(prefix="/collections", tags=["collection"])


# Collection management endpoints


@router.get(
    "",
    summary="List all collections",
    description="Retrieve a list of all collection names in the system.",
)
def list_collections(backend: BackendDependency) -> list[CollectionName]:
    """List all collections."""
    return backend.list_collections()


@router.get(
    "/{collection}",
    responses={404: {"model": NotFoundError}},
    summary="Get collection details",
    description=(
        "Retrieve details for a specific collection, including all its versions "
        "and resolutions."
    ),
)
def get_collection(
    backend: BackendDependency,
    collection: CollectionName,
) -> Collection:
    """Get collection details with all versions and resolutions."""
    return backend.get_collection(name=collection)


@router.post(
    "/{collection}",
    responses={
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
    },
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Create a new collection",
    description="Create a new collection with the specified name.",
)
def create_collection(
    backend: BackendDependency,
    collection: CollectionName,
) -> ResourceOperationStatus:
    """Create a new collection."""
    try:
        backend.create_collection(name=collection)
        return ResourceOperationStatus(
            success=True, name=collection, operation=CRUDOperation.CREATE
        )
    except MatchboxCollectionAlreadyExists as e:
        raise HTTPException(
            status_code=409,
            detail=ResourceOperationStatus(
                success=False,
                name=collection,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ),
        ) from e


@router.delete(
    "/{collection}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Delete a collection",
    description="Delete a collection and all its versions. Requires confirmation.",
)
def delete_collection(
    backend: BackendDependency,
    collection: CollectionName,
    certain: Annotated[
        bool, Query(description="Confirm deletion of the collection")
    ] = False,
) -> ResourceOperationStatus:
    """Delete a collection."""
    try:
        backend.delete_collection(name=collection, certain=certain)
        return ResourceOperationStatus(
            success=True, name=collection, operation=CRUDOperation.DELETE
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxDeletionNotConfirmed,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=collection,
                operation=CRUDOperation.DELETE,
                details=str(e),
            ),
        ) from e


# Run management endpoints


@router.get(
    "/{collection}/runs/{run_id}",
    responses={404: {"model": NotFoundError}},
    summary="Get specific run",
    description="Retrieve details for a specific run within a collection.",
)
def get_run(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
) -> Run:
    """Get a specific run."""
    return backend.get_run(collection=collection, run_id=run_id)


@router.post(
    "/{collection}/runs",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
    },
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Create a new run",
    description="Create a new run within the specified collection.",
)
def create_run(
    backend: BackendDependency,
    collection: CollectionName,
) -> Run:
    """Create a new run in a collection."""
    try:
        return backend.create_run(collection=collection)
    except MatchboxCollectionNotFoundError as e:
        raise HTTPException(
            status_code=409,
            detail=ResourceOperationStatus(
                success=False,
                name=collection,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ),
        ) from e


@router.patch(
    "/{collection}/runs/{run_id}/mutable",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Change run mutability",
    description="Set whether a run can be modified.",
)
def set_run_mutable(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    mutable: Annotated[bool, Body(description="Mutability setting")],
) -> ResourceOperationStatus:
    """Set run mutability."""
    try:
        backend.set_run_mutable(collection=collection, run_id=run_id, mutable=mutable)
        return ResourceOperationStatus(
            success=True,
            name=run_id,
            operation=CRUDOperation.UPDATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=run_id,
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.patch(
    "/{collection}/runs/{run_id}/default",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Change default run",
    description="Set whether a run is the default for its collection.",
)
def set_run_default(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    default: Annotated[bool, Body(description="Default setting")],
) -> ResourceOperationStatus:
    """Set run as default."""
    try:
        backend.set_run_default(collection=collection, run_id=run_id, default=default)
        return ResourceOperationStatus(
            success=True,
            name=run_id,
            operation=CRUDOperation.UPDATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=run_id,
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.delete(
    "/{collection}/runs/{run_id}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Delete a run",
    description="Delete a run and all its resolutions. Requires confirmation.",
)
def delete_run(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    certain: Annotated[bool, Query(description="Confirm deletion of the run")] = False,
) -> ResourceOperationStatus:
    """Delete a run."""
    try:
        backend.delete_run(collection=collection, run_id=run_id, certain=certain)
        return ResourceOperationStatus(
            success=True, name=run_id, operation=CRUDOperation.DELETE
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
        MatchboxDeletionNotConfirmed,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=run_id,
                operation=CRUDOperation.DELETE,
                details=str(e),
            ),
        ) from e


# Resolution management


@router.post(
    "/{collection}/runs/{run_id}/resolutions/{resolution_name}",
    responses={
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Create a resolution",
    description="Create a new resolution (model or source) in the specified run.",
)
def create_resolution(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution_name: ResolutionName,
    resolution: Resolution,
) -> ResourceOperationStatus:
    """Create a resolution (model or source)."""
    try:
        backend.create_resolution(
            resolution=resolution,
            path=ResolutionPath(
                name=resolution_name, collection=collection, run=run_id
            ),
        )
        return ResourceOperationStatus(
            success=True,
            name=resolution_name,
            operation=CRUDOperation.CREATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
    ):
        raise
    except MatchboxResolutionAlreadyExists as e:
        raise HTTPException(
            status_code=409,
            detail=ResourceOperationStatus(
                success=False,
                name=resolution_name,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=resolution_name,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{collection}/runs/{run_id}/resolutions/{resolution}",
    responses={404: {"model": NotFoundError}},
    summary="Get a resolution",
    description="Retrieve a specific resolution (model or source) from the backend.",
)
def get_resolution(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ResolutionName,
    validate_type: ResolutionType | None = None,
) -> Resolution:
    """Get a resolution (model or source) from the backend."""
    return backend.get_resolution(
        path=ResolutionPath(collection=collection, run=run_id, name=resolution),
        validate=validate_type,
    )


@router.delete(
    "/{collection}/runs/{run_id}/resolutions/{resolution}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Delete a resolution",
    description="Delete a resolution from the backend. Requires confirmation.",
)
def delete_resolution(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ResolutionName,
    certain: Annotated[
        bool, Query(description="Confirm deletion of the resolution")
    ] = False,
) -> ResourceOperationStatus:
    """Delete a resolution from the backend."""
    try:
        backend.delete_resolution(
            name=ResolutionPath(collection=collection, run=run_id, name=resolution),
            certain=certain,
        )
        return ResourceOperationStatus(
            success=True,
            name=resolution,
            operation=CRUDOperation.DELETE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
        MatchboxDeletionNotConfirmed,
        MatchboxResolutionNotFoundError,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=resolution,
                operation=CRUDOperation.DELETE,
                details=str(e),
            ),
        ) from e


@router.post(
    "/{collection}/runs/{run_id}/resolutions/{resolution}/data",
    responses={404: {"model": NotFoundError}},
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Set resolution data",
    description="Create an upload task for source hashes or model results.",
)
def set_data(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ModelResolutionName,
    validate_type: ResolutionType | None = None,
) -> UploadStatus:
    """Create an upload task for source hashes or model results."""
    # Get resolution from the specified run
    resolution_path = ResolutionPath(collection=collection, run=run_id, name=resolution)
    resolution = backend.get_resolution(path=resolution_path, validate=validate_type)

    if resolution.resolution_type == ResolutionType.SOURCE:
        upload_id = upload_tracker.add_source(path=resolution_path)
        return upload_tracker.get(upload_id=upload_id).status

    upload_id = upload_tracker.add_model(path=resolution_path)
    return upload_tracker.get(upload_id=upload_id).status


@router.get(
    "/{collection}/runs/{run_id}/resolutions/{resolution}/data",
    responses={404: {"model": NotFoundError}},
    summary="Get resolution results",
    description="Download results for a model as a parquet file.",
)
def get_results(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ModelResolutionName,
) -> ParquetResponse:
    """Download results for a model as a parquet file."""
    res = backend.get_model_data(
        path=ResolutionPath(collection=collection, run=run_id, name=resolution)
    )

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@router.patch(
    "/{collection}/runs/{run_id}/resolutions/{resolution}/truth",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Set resolution truth",
    description="Set truth data for a resolution.",
)
def set_truth(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ModelResolutionName,
    truth: Annotated[int, Body(ge=0, le=100)],
) -> ResourceOperationStatus:
    """Set truth data for a resolution."""
    try:
        backend.set_model_truth(
            path=ResolutionPath(collection=collection, run=run_id, name=resolution),
            truth=truth,
        )
        return ResourceOperationStatus(
            success=True,
            name=resolution,
            operation=CRUDOperation.UPDATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
        MatchboxResolutionNotFoundError,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=resolution,
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{collection}/runs/{run_id}/resolutions/{resolution}/truth",
    responses={404: {"model": NotFoundError}},
    summary="Get resolution truth",
    description="Get truth data for a resolution.",
)
def get_truth(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ModelResolutionName,
) -> float:
    """Get truth data for a resolution."""
    return backend.get_model_truth(
        path=ResolutionPath(collection=collection, run=run_id, name=resolution)
    )
