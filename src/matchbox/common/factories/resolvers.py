"""Factory helpers for resolver testkits."""

import json
from collections.abc import Iterable
from typing import Any

import polars as pl
from faker import Faker
from pydantic import BaseModel, ConfigDict

from matchbox.client.dags import DAG
from matchbox.client.models import Model
from matchbox.client.queries import Query
from matchbox.client.resolvers import (
    Components,
    ComponentsSettings,
    Resolver,
)
from matchbox.common.dtos import ResolverResolutionName, SourceResolutionName
from matchbox.common.factories.entities import ClusterEntity, SourceEntity
from matchbox.common.factories.models import ModelTestkit, model_factory
from matchbox.common.factories.sources import linked_sources_factory
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

    def query(self) -> Query:
        """Thin wrapper to Query this testkit's Sources via its Resolver."""
        return self.resolver.query()

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
    dag: DAG | None = None,
    inputs: Iterable[ModelTestkit] | None = None,
    true_entities: Iterable[SourceEntity] | None = None,
    name: ResolverResolutionName | None = None,
    description: str | None = None,
    seed: int = 42,
) -> ResolverTestkit:
    """Generate a complete resolver testkit.

    Allows autoconfiguration with minimal settings, or more nuanced control.

    Can either be used to generate a resolver in a pipeline, interconnected with
    existing testkit objects, or generate a standalone resolver with random data.

    Args:
        dag: DAG containing this resolver.
            Inferred from the first input testkit if not provided.
            A default DAG is created when inputs are also absent.
        inputs: An iterable of ModelTestkit objects to use as resolver inputs.
            If None, a single default deduper model testkit is created automatically.
            All inputs must belong to the same DAG.
        true_entities: Ground truth SourceEntity objects used to generate the
            expected cluster assignments. If None, the resolver testkit will have
            no expected entities.
        name: Name of the resolver. Defaults to a randomly generated word suffixed
            with '_resolver'.
        description: Description of the resolver.
        seed: Random seed for reproducibility.

    Returns:
        ResolverTestkit: A resolver testkit with generated assignments and expected
            entities.

    Raises:
        TypeError: If any element of inputs is not a ModelTestkit.
        ValueError: If inputs belong to different DAGs.
    """
    if inputs is None:
        dag = dag or DAG(name="collection")
        dag.run = dag.run or 1
        linked = linked_sources_factory(dag=dag, seed=seed)
        default_model = model_factory(
            left_testkit=linked.sources["crn"],
            true_entities=tuple(linked.true_entities),
            seed=seed,
        ).fake_run()
        inputs = (default_model,)
        source_names: set[SourceResolutionName] = set(linked.sources.keys())
    else:
        inputs = tuple(inputs)
        source_names: set[SourceResolutionName] = set()
        for testkit in inputs:
            if not isinstance(testkit, ModelTestkit):
                raise TypeError("resolver_factory inputs must be ModelTestkit.")
            source_names.update(testkit.model.sources)
        dag = dag or inputs[0].model.dag

    input_map: dict[str, ModelTestkit] = {}
    for testkit in inputs:
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
