"""Utilities for querying model results from the PostgreSQL backend."""

from typing import NamedTuple

from sqlalchemy import select

from matchbox.common.dtos import ModelConfig, ModelType, QueryConfig, StepType
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    StepFrom,
    Steps,
)


class SourceInfo(NamedTuple):
    """Information about a model's sources."""

    left: int
    right: int | None
    left_ancestors: set[int]
    right_ancestors: set[int] | None


def _get_model_parents(step_id: int) -> tuple[bytes, bytes | None]:
    """Get the model's immediate parent models."""
    parent_query = (
        select(Steps.step_id, Steps.type)
        .join(StepFrom, Steps.step_id == StepFrom.parent)
        .where(StepFrom.child == step_id)
        .where(StepFrom.level == 1)
    )

    with MBDB.get_session() as session:
        parents = session.execute(parent_query).fetchall()

    if len(parents) == 1:
        return parents[0][0], None
    elif len(parents) == 2:
        p1, p2 = parents
        p1_id, p1_type = p1
        p2_id, p2_type = p2
        # Put source first if it exists
        if p1_type == StepType.SOURCE:
            return p1_id, p2_id
        elif p2_type == StepType.SOURCE:
            return p2_id, p1_id
        # Both models, maintain original order
        return p1_id, p2_id
    else:
        raise ValueError(f"Model has unexpected number of parents: {len(parents)}")


def _get_source_info(step_id: int) -> SourceInfo:
    """Get source steps and their ancestry information."""
    left_id, right_id = _get_model_parents(step_id=step_id)

    with MBDB.get_session() as session:
        left = session.get(Steps, left_id)
        right = session.get(Steps, right_id) if right_id else None

        left_ancestors = {left_id} | {m.step_id for m in left.ancestors}
        if right:
            right_ancestors = {right_id} | {m.step_id for m in right.ancestors}
        else:
            right_ancestors = None

    return SourceInfo(
        left=left_id,
        right=right_id,
        left_ancestors=left_ancestors,
        right_ancestors=right_ancestors,
    )


def get_model_config(step: Steps) -> ModelConfig:
    """Get metadata for a model step."""
    if step.type != StepType.MODEL:
        raise ValueError("Expected step of type model")

    source_info: SourceInfo = _get_source_info(step_id=step.step_id)

    with MBDB.get_session() as session:
        left = session.get(Steps, source_info.left)
        right = session.get(Steps, source_info.right) if source_info.right else None

        return ModelConfig(
            type=ModelType.DEDUPER if source_info.right is None else ModelType.LINKER,
            model_class="",
            model_settings="{}",
            left_query=QueryConfig(source_steps=(left.name,)),
            right_query=(
                QueryConfig(source_steps=(right.name,)) if source_info.right else None
            ),
        )
