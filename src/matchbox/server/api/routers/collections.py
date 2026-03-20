"""Collection and step API routes for the Matchbox server."""

import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    Query,
    UploadFile,
    status,
)
from pyarrow import ArrowInvalid
from pyarrow import parquet as pq

from matchbox.common.arrow import (
    SCHEMA_CLUSTERS,
    SCHEMA_INDEX,
    SCHEMA_MODEL_EDGES,
    table_to_buffer,
)
from matchbox.common.dtos import (
    Collection,
    CollectionName,
    CRUDOperation,
    ErrorResponse,
    GroupName,
    ModelStepPath,
    PermissionGrant,
    PermissionType,
    ResolverStepPath,
    ResourceOperationStatus,
    Run,
    RunID,
    Step,
    StepName,
    StepPath,
    StepType,
    UploadInfo,
)
from matchbox.common.exceptions import (
    MatchboxServerFileError,
    MatchboxStepTypeError,
)
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    RequireCollectionAdmin,
    RequireCollectionRead,
    RequireCollectionWrite,
    SettingsDependency,
    UploadTrackerDependency,
)
from matchbox.server.uploads import (
    file_to_s3,
    process_upload,
    process_upload_celery,
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
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionRead)],
    summary="Get collection details",
    description=(
        "Retrieve details for a specific collection, including all its versions "
        "and steps."
    ),
)
def get_collection(
    backend: BackendDependency,
    collection: CollectionName,
) -> Collection:
    """Get collection details with all versions and steps."""
    return backend.get_collection(name=collection)


@router.post(
    "/{collection}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    status_code=status.HTTP_201_CREATED,
    summary="Create a new collection",
    description="Create a new collection with the specified name.",
)
def create_collection(
    backend: BackendDependency,
    collection: CollectionName,
    permissions: list[PermissionGrant],
) -> ResourceOperationStatus:
    """Create a new collection."""
    backend.create_collection(
        name=collection,
        permissions=permissions,
    )
    return ResourceOperationStatus(
        success=True,
        target=f"Collection {collection}",
        operation=CRUDOperation.CREATE,
    )


@router.delete(
    "/{collection}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionWrite)],
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
    backend.delete_collection(name=collection, certain=certain)
    return ResourceOperationStatus(
        success=True,
        target=f"Collection {collection}",
        operation=CRUDOperation.DELETE,
    )


# Collection permissions endpoints


@router.get(
    "/{collection}/permissions",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionAdmin)],
)
def get_permissions(
    backend: BackendDependency,
    collection: CollectionName,
) -> list[PermissionGrant]:
    """Get permissions for a collection resource."""
    return backend.get_permissions(collection)


@router.post(
    "/{collection}/permissions",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionAdmin)],
)
def grant_permission(
    backend: BackendDependency,
    collection: CollectionName,
    grant: PermissionGrant,
) -> ResourceOperationStatus:
    """Grant a permission on the system resource."""
    backend.grant_permission(grant.group_name, grant.permission, collection)
    return ResourceOperationStatus(
        success=True,
        target=f"{grant.permission} on system for {grant.group_name}",
        operation=CRUDOperation.CREATE,
    )


@router.delete(
    "/{collection}/permissions/{permission}/{group_name}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionAdmin)],
)
def revoke_permission(
    backend: BackendDependency,
    collection: CollectionName,
    permission: PermissionType,
    group_name: GroupName,
) -> ResourceOperationStatus:
    """Revoke a permission on the system resource."""
    backend.revoke_permission(group_name, permission, collection)
    return ResourceOperationStatus(
        success=True,
        target=f"{permission} on system for {group_name}",
        operation=CRUDOperation.DELETE,
    )


# Run management endpoints


@router.get(
    "/{collection}/runs/{run_id}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionRead)],
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
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionWrite)],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new run",
    description="Create a new run within the specified collection.",
)
def create_run(
    backend: BackendDependency,
    collection: CollectionName,
) -> Run:
    """Create a new run in a collection."""
    return backend.create_run(collection=collection)


@router.patch(
    "/{collection}/runs/{run_id}/mutable",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionWrite)],
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
    backend.set_run_mutable(collection=collection, run_id=run_id, mutable=mutable)
    return ResourceOperationStatus(
        success=True,
        target=f"Run {run_id}",
        operation=CRUDOperation.UPDATE,
    )


@router.patch(
    "/{collection}/runs/{run_id}/default",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionWrite)],
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
    backend.set_run_default(collection=collection, run_id=run_id, default=default)
    return ResourceOperationStatus(
        success=True,
        target=f"Run {run_id}",
        operation=CRUDOperation.UPDATE,
    )


@router.delete(
    "/{collection}/runs/{run_id}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionWrite)],
    summary="Delete a run",
    description="Delete a run and all its steps. Requires confirmation.",
)
def delete_run(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    certain: Annotated[bool, Query(description="Confirm deletion of the run")] = False,
) -> ResourceOperationStatus:
    """Delete a run."""
    backend.delete_run(collection=collection, run_id=run_id, certain=certain)
    return ResourceOperationStatus(
        success=True, target=f"Run {run_id}", operation=CRUDOperation.DELETE
    )


# Step management


@router.post(
    "/{collection}/runs/{run_id}/steps/{step_name}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionWrite)],
    status_code=status.HTTP_201_CREATED,
    summary="Create a step",
    description="Create a new step in the specified run.",
)
def create_step(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    step_name: StepName,
    step: Step,
) -> ResourceOperationStatus:
    """Create a step."""
    step_path = StepPath(name=step_name, collection=collection, run=run_id)
    backend.create_step(
        step=step,
        path=step_path,
    )
    return ResourceOperationStatus(
        success=True,
        target=f"Step {step_path}",
        operation=CRUDOperation.CREATE,
    )


@router.get(
    "/{collection}/runs/{run_id}/steps/{step_name}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionRead)],
    summary="Get a step",
    description="Retrieve a specific step from the backend.",
)
def get_step(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    step_name: StepName,
) -> Step:
    """Get a step from the backend."""
    return backend.get_step(
        path=StepPath(collection=collection, run=run_id, name=step_name)
    )


@router.put(
    "/{collection}/runs/{run_id}/steps/{step_name}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionWrite)],
    status_code=status.HTTP_200_OK,
    summary="Update a step",
    description="Update an existing step in the specified run.",
)
def update_step(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    step_name: StepName,
    step: Step,
) -> ResourceOperationStatus:
    """Update an existing step."""
    step_path = StepPath(name=step_name, collection=collection, run=run_id)
    backend.update_step(
        step=step,
        path=step_path,
    )
    return ResourceOperationStatus(
        success=True,
        target=f"Step {step_path}",
        operation=CRUDOperation.UPDATE,
    )


@router.delete(
    "/{collection}/runs/{run_id}/steps/{step_name}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionWrite)],
    summary="Delete a step",
    description="Delete a step from the backend. Requires confirmation.",
)
def delete_step(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    step_name: StepName,
    certain: Annotated[bool, Query(description="Confirm deletion of the step")] = False,
) -> ResourceOperationStatus:
    """Delete a step from the backend."""
    step_path = StepPath(collection=collection, run=run_id, name=step_name)
    backend.delete_step(path=step_path, certain=certain)
    return ResourceOperationStatus(
        success=True,
        target=f"Step {step_path}",
        operation=CRUDOperation.DELETE,
    )


@router.post(
    "/{collection}/runs/{run_id}/steps/{step_name}/data",
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        423: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionWrite)],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Set step data",
    description="Create an upload task for any step data.",
)
def set_data(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
    settings: SettingsDependency,
    background_tasks: BackgroundTasks,
    collection: CollectionName,
    run_id: RunID,
    step_name: StepName,
    file: UploadFile,
) -> ResourceOperationStatus:
    """Create an upload task for any step data."""
    step_path = StepPath(collection=collection, run=run_id, name=step_name)

    # Check if data is locked, lock it if not (raises MatchboxLockError -> 423)
    backend.lock_step_data(path=step_path)

    # Try-except to ensure we release the lock
    try:
        # Validate file extension
        if ".parquet" not in file.filename:
            extension = file.filename.split(".")[-1]
            raise MatchboxServerFileError(
                message=f"Expected .parquet file, got {extension}"
            )

        # pyarrow validates Parquet magic numbers when loading file
        try:
            pq.ParquetFile(file.file)
        except ArrowInvalid as e:
            raise MatchboxServerFileError(
                message=f"Invalid Parquet file: {str(e)}"
            ) from e

        # Get step
        step = backend.get_step(path=step_path)

        # Generate unique upload id
        upload_id = str(uuid.uuid4())

        # Upload to S3
        client = backend.settings.datastore.get_client()
        bucket = backend.settings.datastore.cache_bucket_name
        key = f"{upload_id}.parquet"

        if step.step_type == StepType.SOURCE:
            expected_schema = SCHEMA_INDEX
        elif step.step_type == StepType.MODEL:
            expected_schema = SCHEMA_MODEL_EDGES
        elif step.step_type == StepType.RESOLVER:
            expected_schema = SCHEMA_CLUSTERS
        else:
            raise RuntimeError("Unsupported step type.")

        table = pq.read_table(file.file)
        if not table.schema.equals(expected_schema):
            raise MatchboxServerFileError(
                message=(
                    "Schema mismatch. "
                    f"Expected:\n{expected_schema}\nGot:\n{table.schema}"
                )
            )

        # Upload to object storage (raises MatchboxServerFileError on failure)
        file_to_s3(client=client, bucket=bucket, key=key, file=file)

        match settings.task_runner:
            case "api":
                background_tasks.add_task(
                    process_upload,
                    backend=backend,
                    tracker=upload_tracker,
                    s3_client=client,
                    step_path=step_path,
                    upload_id=upload_id,
                    bucket=bucket,
                    filename=key,
                )
            case "celery":
                process_upload_celery.delay(
                    step_path_json=step_path.model_dump_json(),
                    upload_id=upload_id,
                    bucket=bucket,
                    filename=key,
                )
            case _:
                raise RuntimeError("Unsupported task runner.")

        return ResourceOperationStatus(
            success=True,
            target=f"Step data {step_path}",
            operation=CRUDOperation.CREATE,
            details=upload_id,
        )
    except:
        backend.unlock_step_data(path=step_path)
        raise


@router.get(
    "/{collection}/runs/{run_id}/steps/{step_name}/data/status",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionRead)],
    status_code=status.HTTP_200_OK,
)
def get_upload_status(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
    collection: CollectionName,
    run_id: RunID,
    step_name: StepName,
    upload_id: Annotated[str | None, Query()] = None,
) -> UploadInfo:
    """Get the status of an upload process.

    Optionally looks for error message if given an upload ID.
    """
    error = None
    if upload_id:
        error = upload_tracker.get(upload_id=upload_id)

    step_stage = backend.get_step_stage(
        path=StepPath(collection=collection, run=run_id, name=step_name)
    )

    return UploadInfo(stage=step_stage, error=error)


@router.get(
    "/{collection}/runs/{run_id}/steps/{step_name}/data",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionRead)],
    summary="Get step data",
    description="Download data for a step as a parquet file.",
)
def get_data(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    step_name: StepName,
) -> ParquetResponse:
    """Download data for a step as a parquet file."""
    step_path = StepPath(collection=collection, run=run_id, name=step_name)
    step_dto = backend.get_step(path=step_path)
    if step_dto.step_type == StepType.MODEL:
        res = backend.get_model_data(
            path=ModelStepPath(
                collection=collection,
                run=run_id,
                name=step_name,
            )
        )
    elif step_dto.step_type == StepType.RESOLVER:
        res = backend.get_resolver_data(
            path=ResolverStepPath(
                collection=collection,
                run=run_id,
                name=step_name,
            )
        )
    else:
        raise MatchboxStepTypeError(
            step_name=StepPath(collection=collection, run=run_id, name=step_name),
            step_type=step_dto.step_type,
            expected_step_types=[StepType.MODEL, StepType.RESOLVER],
        )

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())
