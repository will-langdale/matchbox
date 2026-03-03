"""Connected-components resolver methodology."""

from collections.abc import Mapping
from typing import ClassVar

import polars as pl
from pydantic import Field, field_validator

from matchbox.client.resolvers.base import ResolverMethod, ResolverSettings
from matchbox.common.arrow import SCHEMA_CLUSTERS
from matchbox.common.dtos import ResolutionName, ResolverType
from matchbox.common.transform import DisjointSet, threshold_float_to_int


class ComponentsSettings(ResolverSettings):
    """Settings for the Components resolver methodology."""

    thresholds: dict[ResolutionName, int | float] = Field(default_factory=dict)

    @field_validator("thresholds", mode="after")
    @classmethod
    def normalise_thresholds(
        cls, thresholds: dict[ResolutionName, int | float]
    ) -> dict[ResolutionName, int]:
        """Normalise thresholds to backend integer percentages."""
        normalised: dict[ResolutionName, int] = {}
        for input_name, threshold in thresholds.items():
            if isinstance(threshold, bool):
                raise ValueError(
                    "Thresholds must be floats in [0,1] or ints in [0,100]"
                )
            if isinstance(threshold, float):
                normalised[input_name] = threshold_float_to_int(threshold)
                continue
            if isinstance(threshold, int) and 0 <= threshold <= 100:
                normalised[input_name] = threshold
                continue
            raise ValueError("Thresholds must be floats in [0,1] or ints in [0,100]")
        return normalised


class Components(ResolverMethod):
    """Resolver methodology that computes connected components."""

    resolver_type: ClassVar[ResolverType] = ResolverType.COMPONENTS
    settings: ComponentsSettings

    def compute_clusters(
        self,
        model_edges: Mapping[ResolutionName, pl.DataFrame],
        resolver_assignments: Mapping[ResolutionName, pl.DataFrame],
    ) -> pl.DataFrame:
        """Compute connected components from model edges and resolver assignments."""
        djs = DisjointSet[int]()
        all_nodes: set[int] = set()

        for model_name, edges in model_edges.items():
            if edges.height == 0:
                continue

            threshold = self.settings.thresholds.get(model_name, 0)
            filtered_edges = edges.filter(pl.col("probability") >= threshold)
            for left_id, right_id in filtered_edges.select(
                "left_id", "right_id"
            ).iter_rows():
                left = int(left_id)
                right = int(right_id)
                all_nodes.update((left, right))
                djs.union(left, right)

        for assignments in resolver_assignments.values():
            if assignments.height == 0:
                continue
            assignment_df = assignments.select("parent_id", "child_id").cast(
                pl.Schema(SCHEMA_CLUSTERS)
            )

            grouped_nodes = assignment_df.group_by("parent_id").agg(
                pl.col("child_id").unique().sort()
            )

            for node_ids in grouped_nodes["child_id"]:
                component = [int(node_id) for node_id in node_ids]
                if not component:
                    continue
                all_nodes.update(component)
                anchor = component[0]
                djs.add(anchor)
                for other in component[1:]:
                    djs.union(anchor, other)

        for node_id in all_nodes:
            djs.add(node_id)

        components = [sorted(component) for component in djs.get_components()]
        components = sorted(components, key=lambda values: values[0])

        rows: list[dict[str, int]] = []
        for parent_id, component in enumerate(components, start=1):
            rows.extend(
                {"parent_id": parent_id, "child_id": node_id} for node_id in component
            )

        if not rows:
            return pl.from_arrow(SCHEMA_CLUSTERS.empty_table())

        return pl.DataFrame(rows).cast(pl.Schema(SCHEMA_CLUSTERS))
