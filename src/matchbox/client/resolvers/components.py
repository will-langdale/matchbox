"""Connected-components resolver methodology."""

from collections.abc import Iterable, Mapping
from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from matchbox.client.resolvers.base import ResolverMethod, ResolverSettings
from matchbox.common.arrow import SCHEMA_CLUSTERS
from matchbox.common.dtos import ModelStepName, ResolverType
from matchbox.common.transform import DisjointSet


class ComponentsSettings(ResolverSettings):
    """Settings for the Components resolver methodology."""

    thresholds: dict[ModelStepName, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default_factory=dict
    )

    def validate_inputs(self, model_names: Iterable[ModelStepName]) -> None:  # noqa: D102
        if missing := set(model_names) - set(self.thresholds.keys()):
            raise RuntimeError(f"Missing thresholds for models: {missing}")


class Components(ResolverMethod):
    """Resolver methodology that computes connected components."""

    resolver_type: ClassVar[ResolverType] = ResolverType.COMPONENTS
    settings: ComponentsSettings

    def compute_clusters(  # noqa: D102
        self, model_edges: Mapping[ModelStepName, pl.DataFrame]
    ) -> pl.DataFrame:
        self.settings.validate_inputs(model_edges.keys())

        djs = DisjointSet[int]()

        for model_name, edges in model_edges.items():
            if edges.height == 0:
                continue

            threshold = self.settings.thresholds[model_name]
            filtered_edges = edges.filter(pl.col("score") >= threshold)

            for left_id, right_id in filtered_edges.select(
                "left_id", "right_id"
            ).iter_rows():
                djs.union(left_id, right_id)

        rows: list[dict[str, int]] = []
        for parent_id, component in enumerate(djs.get_components(), start=1):
            rows.extend(
                {"parent_id": parent_id, "child_id": node_id} for node_id in component
            )

        if not rows:
            return pl.from_arrow(SCHEMA_CLUSTERS.empty_table())

        return pl.DataFrame(rows).cast(pl.Schema(SCHEMA_CLUSTERS))
