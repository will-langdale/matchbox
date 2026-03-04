"""Factory helpers for resolver testkits."""

import json
from collections.abc import Iterable
from typing import Any

import polars as pl
from faker import Faker
from pydantic import BaseModel, ConfigDict

from matchbox.client.dags import DAG
from matchbox.client.models import Model
from matchbox.client.resolvers import (
    Components,
    ComponentsSettings,
    Resolver,
)
from matchbox.common.dtos import ResolverResolutionName
from matchbox.common.factories.entities import (
    ClusterEntity,
    SourceEntity,
)
from matchbox.common.factories.models import ModelTestkit, model_factory
from matchbox.common.transform import threshold_int_to_float


class ResolverTestkit(BaseModel):
    """Resolver plus local expected data for tests."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resolver: Resolver
    assignments: pl.DataFrame
    entities: tuple[ClusterEntity, ...]

    @property
    def name(self) -> str:
        """Return resolver name."""
        return self.resolver.name

    def into_dag(self) -> dict[str, Any]:
        """Return kwargs for explicit DAG insertion."""
        config = self.resolver.config
        return {
            "name": self.resolver.name,
            "inputs": list(config.inputs),
            "resolver_class": config.resolver_class,
            "resolver_settings": json.loads(config.resolver_settings),
            "description": self.resolver.description,
        }


def resolver_factory(
    *,
    dag: DAG,
    inputs: Iterable[ModelTestkit] | None = None,
    true_entities: Iterable[SourceEntity] | None = None,
    name: ResolverResolutionName | None = None,
    description: str | None = None,
    seed: int = 42,
) -> ResolverTestkit:
    """Build a detached resolver testkit and local expected entities.

    Defaults:
    - `inputs=None`: create a single default model testkit.
    - Components resolver with thresholds inferred from each model threshold.
    """
    if inputs is None:
        default_model = model_factory(dag=dag, seed=seed).fake_run()
        for source in default_model.left_query.sources:
            dag._add_step(source)  # noqa: SLF001
        if default_model.right_query is not None:
            for source in default_model.right_query.sources:
                dag._add_step(source)  # noqa: SLF001
        dag._add_step(default_model.model)  # noqa: SLF001
        inputs = (default_model,)

    input_map: dict[str, ModelTestkit] = {}
    for testkit in inputs:
        if not isinstance(testkit, ModelTestkit):
            raise TypeError("resolver_factory inputs must be ModelTestkit.")
        input_map.setdefault(testkit.name, testkit)

    resolver_inputs: list[Model] = []
    for testkit in input_map.values():
        if testkit.model.dag != dag:
            raise ValueError("Cannot mix DAGs when building a resolver testkit.")
        if testkit.model.results is None:
            testkit.fake_run()
        resolver_inputs.append(testkit.model)

    resolver_settings = ComponentsSettings(
        thresholds={
            testkit.name: threshold_int_to_float(testkit.threshold)
            for testkit in input_map.values()
        }
    )

    generator = Faker()
    generator.seed_instance(seed)
    resolver = Resolver(
        dag=dag,
        name=name or f"{generator.unique.word()}_resolver",
        inputs=resolver_inputs,
        resolver_class=Components,
        resolver_settings=resolver_settings,
        description=description,
    )
    assignments = resolver.run()

    source_names = tuple(sorted(resolver.sources))
    source_entities = tuple(true_entities or ())
    entities = tuple(
        projected
        for entity in source_entities
        if (projected := entity.to_cluster_entity(*source_names)) is not None
    )

    return ResolverTestkit(
        resolver=resolver,
        assignments=assignments,
        entities=entities,
    )
