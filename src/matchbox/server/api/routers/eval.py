"""Resolutions API routes for the Matchbox server."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, status

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendUnprocessableType,
    ModelResolutionName,
    ResolutionOperationStatus,
    UnprocessableError,
)
from matchbox.common.eval import Judgement, ModelComparison
from matchbox.common.exceptions import MatchboxDataNotFound, MatchboxUserNotFoundError
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
)

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post(
    "/judgements",
    responses={422: {"model": UnprocessableError}},
    status_code=status.HTTP_201_CREATED,
)
async def insert_judgement(
    backend: BackendDependency,
    judgement: Judgement,
):
    """Submit judgement from human evaluator."""
    try:
        backend.insert_judgement(judgement=judgement)
        return Response(status_code=status.HTTP_201_CREATED)
    except MatchboxDataNotFound as e:
        raise HTTPException(
            status_code=422,
            detail=UnprocessableError(
                details=str(e), entity=BackendUnprocessableType.CLUSTERS
            ).model_dump(),
        ) from e
    except MatchboxUserNotFoundError as e:
        raise HTTPException(
            status_code=422,
            detail=UnprocessableError(
                details=str(e), entity=BackendUnprocessableType.USERS
            ).model_dump(),
        ) from e


@router.get(
    "/judgements",
)
async def get_judgements(backend: BackendDependency) -> ParquetResponse:
    """Retrieve all judgements from human evaluators."""
    buffer = table_to_buffer(backend.get_judgements())
    return ParquetResponse(buffer.getvalue())


@router.get(
    "/compare",
)
async def compare_models(
    backend: BackendDependency,
    resolutions: Annotated[
        list[ModelResolutionName],
        Query(description="The resolution names for the models to compare."),
    ],
) -> ModelComparison:
    """Return comparison of selected models."""
    return {
        "companieshouse_datahub_deterministic": (1, 0.8),
        "companieshouse_datahub_probabilistic": (0.8, 0.9),
    }


@router.get(
    "/samples",
)
async def sample(
    backend: BackendDependency, n: int, resolution: ModelResolutionName, user_id: int
) -> ResolutionOperationStatus:
    """Sample n cluster to validate."""
