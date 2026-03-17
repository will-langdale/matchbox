"""Resolver nodes and methodology registry for client-side execution."""

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import pyarrow as pa

from matchbox.client.models.models import Model
from matchbox.client.queries import Query
from matchbox.client.resolvers.base import ResolverMethod, ResolverSettings
from matchbox.client.resolvers.components import Components
from matchbox.client.steps import Step, post_run
from matchbox.common.arrow import SCHEMA_CLUSTERS, check_schema_subset
from matchbox.common.dtos import (
    Resolution,
    ResolutionName,
    ResolutionType,
    ResolverConfig,
    ResolverResolutionName,
    ResolverResolutionPath,
    SourceResolutionName,
)
from matchbox.common.exceptions import MatchboxResolutionTypeError
from matchbox.common.hash import hash_clusters
from matchbox.common.logging import logger, profile_time

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.sources import Source
else:
    DAG = Any
    Source = Any

_RESOLVER_CLASSES: dict[str, type[ResolverMethod]] = {}


def add_resolver_class(resolver_class: type[ResolverMethod]) -> None:
    """Register a resolver methodology class."""
    if not issubclass(resolver_class, ResolverMethod):
        raise ValueError("The argument is not a proper subclass of ResolverMethod.")
    _RESOLVER_CLASSES[resolver_class.__name__] = resolver_class
    logger.debug(f"Registered resolver class: {resolver_class.__name__}")


add_resolver_class(Components)


class Resolver(Step):
    """Client-side node that computes clusters from model and resolver inputs."""

    _local_data_schema: ClassVar[pa.Schema] = SCHEMA_CLUSTERS

    def __init__(
        self,
        dag: DAG,
        name: ResolverResolutionName,
        inputs: Iterable[Model],
        resolver_class: type[ResolverMethod] | str,
        resolver_settings: ResolverSettings | dict[str, Any],
        description: str | None = None,
    ) -> None:
        """Create a resolver node that computes clusters from its inputs."""
        super().__init__(
            dag=dag, name=ResolverResolutionName(name), description=description
        )

        deduped_inputs: list[Model] = []
        seen_names: set[str] = set()
        for node in inputs:
            if not isinstance(node, Model):
                raise MatchboxResolutionTypeError(
                    resolution_name=getattr(node, "name", node),
                    expected_resolution_types=[ResolutionType.MODEL],
                )
            if node.name in seen_names:
                continue
            seen_names.add(node.name)
            deduped_inputs.append(node)
        self.inputs = tuple(deduped_inputs)

        if len(self.inputs) < 1:
            raise ValueError("Resolver needs at least one input")

        if isinstance(resolver_class, str):
            self.resolver_class: type[ResolverMethod] = _RESOLVER_CLASSES[
                resolver_class
            ]
        else:
            self.resolver_class = resolver_class

        self.resolver_instance = self.resolver_class(settings=resolver_settings)

        if isinstance(resolver_settings, dict):
            SettingsClass = self.resolver_instance.__annotations__["settings"]
            self.resolver_settings = SettingsClass(**resolver_settings)
        else:
            self.resolver_settings = resolver_settings

    @property
    def results(self) -> pl.DataFrame | None:
        """The locally computed cluster assignments. Alias for local_data."""
        return self._local_data

    @results.setter
    def results(self, value: pl.DataFrame | None) -> None:
        self._local_data = value

    @property
    @post_run
    def results_eval(self) -> pl.DataFrame:
        """Get mapping of result clusters to leaf IDs from the server."""
        leaf_id_mappings: list[pl.DataFrame] = []

        for model in self.inputs:
            error_str = (
                f"Model {model.name} has no leaf data. Re-run with low_memory=False"
            )

            if (left := model.left_query.leaf_id) is None:
                raise RuntimeError(error_str)
            leaf_id_mappings.append(left)

            if model.right_query:
                if (right := model.left_query.leaf_id) is None:
                    raise RuntimeError(error_str)
                leaf_id_mappings.append(right)

        all_mappings = pl.concat(leaf_id_mappings)

        return (
            self.results.join(all_mappings, left_on="child_id", right_on="id")
            .select("parent_id", "leaf_id")
            .rename({"parent_id": "root", "leaf_id": "leaf"})
            .unique()
        )

    @property
    def config(self) -> ResolverConfig:
        """Generate config DTO from Resolver."""
        return ResolverConfig(
            resolver_class=self.resolver_class.__name__,
            resolver_settings=self.resolver_settings.model_dump_json(),
            inputs=tuple(node.name for node in self.inputs),
        )

    @property
    def sources(self) -> set[SourceResolutionName]:
        """Set of source names upstream of this node."""
        upstream: set[SourceResolutionName] = set()
        for node in self.inputs:
            upstream.update(node.sources)
        return upstream

    @property
    def resolution_path(self) -> ResolverResolutionPath:
        """Return resolver path."""
        return ResolverResolutionPath(
            collection=self.dag.name,
            run=self.dag.run,
            name=self.name,
        )

    @profile_time(attr="name")
    def compute_clusters(
        self, model_edges: Mapping[ResolutionName, pl.DataFrame]
    ) -> pl.DataFrame:
        """Delegate cluster computation to the configured resolver instance."""
        return self.resolver_instance.compute_clusters(model_edges=model_edges)

    @profile_time(attr="name")
    def run(self) -> pl.DataFrame:
        """Run the resolver and materialise cluster assignments."""
        model_edges: dict[ResolutionName, pl.DataFrame] = {}

        for node in self.inputs:
            if node.results is None:
                raise ValueError(
                    f"Resolver input '{node.name}' has no local results. "
                    "Run or download upstream nodes before running this resolver."
                )
            model_edges[node.name] = node.results

        self._local_data = self.compute_clusters(model_edges=model_edges)
        return self._local_data

    @post_run
    def _fingerprint(self) -> bytes:
        """Compute resolver fingerprint from semantic cluster membership."""
        check_schema_subset(
            expected=self._local_data_schema, actual=self._local_data.to_arrow().schema
        )
        return hash_clusters(self._local_data.to_arrow())

    @post_run
    def to_resolution(self) -> Resolution:
        """Convert to Resolution for API calls."""
        return Resolution(
            description=self.description,
            resolution_type=ResolutionType.RESOLVER,
            config=self.config,
            fingerprint=self._fingerprint(),
        )

    @classmethod
    def from_resolution(
        cls,
        resolution: Resolution,
        resolution_name: str,
        dag: DAG,
        **kwargs: Any,
    ) -> "Resolver":
        """Reconstruct from Resolution."""
        if resolution.resolution_type != ResolutionType.RESOLVER:
            raise ValueError("Resolution must be of type 'resolver'")

        return cls(
            dag=dag,
            name=ResolverResolutionName(resolution_name),
            description=resolution.description,
            inputs=[dag.nodes[name] for name in resolution.config.inputs],
            resolver_class=resolution.config.resolver_class,
            resolver_settings=json.loads(resolution.config.resolver_settings),
        )

    def query(self, *sources: Source, **kwargs: Any) -> Query:
        """Create a query rooted at this resolver."""
        if not sources:
            sources = tuple(self.dag.get_source(name) for name in sorted(self.sources))
        return Query(*sources, resolver=self, dag=self.dag, **kwargs)
