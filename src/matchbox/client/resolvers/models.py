"""Resolver nodes and methodology registry for client-side execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import polars as pl

from matchbox.client import _handler
from matchbox.client.models.models import Model
from matchbox.client.queries import Query
from matchbox.client.resolvers.base import ResolverMethod, ResolverSettings
from matchbox.client.resolvers.components import Components, ComponentsSettings
from matchbox.common.arrow import SCHEMA_CLUSTERS, check_schema
from matchbox.common.dtos import (
    Resolution,
    ResolutionName,
    ResolutionType,
    ResolverConfig,
    ResolverResolutionName,
    ResolverResolutionPath,
    ResolverType,
    SourceResolutionName,
)
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.hash import hash_arrow_table
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


def get_resolver_class(class_name: str) -> type[ResolverMethod]:
    """Retrieve a resolver methodology class by name."""
    try:
        return _RESOLVER_CLASSES[class_name]
    except KeyError as e:
        raise ValueError(f"Unknown resolver class: {class_name}") from e


add_resolver_class(Components)


class Resolver:
    """Client-side node that computes clusters from model and resolver inputs."""

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
        self.dag = dag
        self.name = ResolverResolutionName(name)
        deduped_inputs: list[Model] = []
        seen_names: set[str] = set()
        for node in inputs:
            if not isinstance(node, Model):
                raise ValueError(
                    f"Resolver input '{getattr(node, 'name', node)}' is not a model"
                )
            if node.name in seen_names:
                continue
            seen_names.add(node.name)
            deduped_inputs.append(node)
        self.inputs = tuple(deduped_inputs)
        self.description = description

        if len(self.inputs) < 1:
            raise ValueError("Resolver needs at least one input")

        if isinstance(resolver_class, str):
            self.resolver_class = get_resolver_class(resolver_class)
        else:
            self.resolver_class = resolver_class

        self.resolver_instance = self.resolver_class(settings=resolver_settings)
        self.resolver_type = self._infer_resolver_type(self.resolver_class)
        self.resolver_settings = self.resolver_instance.settings

        self.resolver_instance.settings = self.resolver_settings
        self._normalise_components_settings()

        self.results: pl.DataFrame | None = None

    @staticmethod
    def _infer_resolver_type(
        resolver_class: type[ResolverMethod],
    ) -> ResolverType:
        """Infer resolver type from resolver class metadata."""
        resolver_type = getattr(resolver_class, "resolver_type", None)
        if resolver_type is None:
            if issubclass(resolver_class, Components):
                return ResolverType.COMPONENTS
            raise ValueError(
                f"Resolver class '{resolver_class.__name__}' must define resolver_type"
            )
        return ResolverType(resolver_type)

    def _normalise_components_settings(self) -> None:
        """Ensure Components settings are aligned with configured inputs."""
        if self.resolver_type != ResolverType.COMPONENTS:
            return

        if not isinstance(self.resolver_settings, ComponentsSettings):
            self.resolver_settings = ComponentsSettings.model_validate(
                self.resolver_settings.model_dump(mode="json")
            )

        input_names = tuple(node.name for node in self.inputs)
        threshold_input = dict(self.resolver_settings.thresholds)

        extra_thresholds = [name for name in threshold_input if name not in input_names]
        if extra_thresholds:
            raise ValueError(
                "Thresholds were provided for unknown resolver inputs: "
                f"{extra_thresholds}"
            )

        for node_name in input_names:
            threshold_input.setdefault(node_name, 0)

        self.resolver_settings = ComponentsSettings(thresholds=threshold_input)
        self.resolver_instance.settings = self.resolver_settings

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
        self,
        model_edges: Mapping[ResolutionName, pl.DataFrame],
        resolver_assignments: Mapping[ResolutionName, pl.DataFrame],
    ) -> pl.DataFrame:
        """Delegate cluster computation to the configured resolver instance."""
        return self.resolver_instance.compute_clusters(
            model_edges=model_edges,
            resolver_assignments=resolver_assignments,
        )

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

        clusters = self.compute_clusters(
            model_edges=model_edges,
            resolver_assignments={},
        )

        self.results = clusters
        return self.results

    def to_resolution(self) -> Resolution:
        """Convert to Resolution for API calls."""
        if self.results is None:
            raise RuntimeError("Resolver must be run before converting to a resolution")

        upload_table = self.results.to_arrow()
        check_schema(expected=SCHEMA_CLUSTERS, actual=upload_table.schema)
        return Resolution(
            description=self.description,
            resolution_type=ResolutionType.RESOLVER,
            config=self.config,
            fingerprint=hash_arrow_table(upload_table),
        )

    @profile_time(attr="name")
    def sync(self) -> None:
        """Send resolver config and assignments to the server."""
        resolution = self.to_resolution()
        log_prefix = f"Sync {self.name}"
        should_upload = False

        try:
            existing_resolution = _handler.get_resolution(path=self.resolution_path)
            logger.info("Found existing resolution", prefix=log_prefix)
        except MatchboxResolutionNotFoundError:
            existing_resolution = None

        if existing_resolution:
            if (existing_resolution.fingerprint == resolution.fingerprint) and (
                existing_resolution.config.parents == resolution.config.parents
            ):
                logger.info("Updating existing resolution", prefix=log_prefix)
                _handler.update_resolution(
                    resolution=resolution,
                    path=self.resolution_path,
                )
            else:
                logger.info(
                    "Update not possible. Deleting existing resolution",
                    prefix=log_prefix,
                )
                _handler.delete_resolution(path=self.resolution_path, certain=True)
                existing_resolution = None

        if not existing_resolution:
            logger.info("Creating new resolution", prefix=log_prefix)
            _handler.create_resolution(resolution=resolution, path=self.resolution_path)
            should_upload = True

        if should_upload:
            if self.results is None:
                raise RuntimeError("Resolver must be run before sync")

            _handler.set_data(
                path=self.resolution_path,
                data=self.results,
            )

        # Refresh local state from backend canonical resolver data.
        self.results = self.download_results()

    def query(self, *sources: Source, **kwargs: Any) -> Query:
        """Create a query rooted at this resolver."""
        if not sources:
            sources = tuple(self.dag.get_source(name) for name in sorted(self.sources))
        return Query(*sources, resolver=self, dag=self.dag, **kwargs)

    def download_results(self) -> pl.DataFrame:
        """Download resolver assignments directly from the resolution data API."""
        table = _handler.get_resolver_data(path=self.resolution_path)
        check_schema(expected=SCHEMA_CLUSTERS, actual=table.schema)
        self.results = pl.from_arrow(table)
        return self.results

    def clear_data(self) -> None:
        """Drop local resolver data."""
        self.results = None

    def delete(self, certain: bool = False) -> bool:
        """Delete resolver and associated data from backend."""
        return _handler.delete_resolution(
            path=self.resolution_path,
            certain=certain,
        ).success
