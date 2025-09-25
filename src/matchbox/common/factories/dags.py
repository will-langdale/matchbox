"""Simplified TestkitDAG that's just a registry of test data."""

from pydantic import BaseModel, ConfigDict, Field

from matchbox.client.dags import DAG
from matchbox.common.dtos import (
    UnqualifiedModelResolutionName,
    UnqualifiedResolutionName,
    UnqualifiedSourceResolutionName,
)
from matchbox.common.factories.models import ModelTestkit
from matchbox.common.factories.sources import LinkedSourcesTestkit, SourceTestkit


class TestkitDAG(BaseModel):
    """DAG test wrapper that's just a registry of test data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The real DAG that handles all logic
    dag: DAG = Field(default_factory=lambda: DAG(name="collection", version="draft"))

    # Just registries of test data - no complex logic
    sources: dict[UnqualifiedSourceResolutionName, SourceTestkit] = {}
    models: dict[UnqualifiedModelResolutionName, ModelTestkit] = {}
    linked: dict[str, LinkedSourcesTestkit] = {}

    def add_source(self, testkit: SourceTestkit | LinkedSourcesTestkit) -> None:
        """Add source(s) to the real DAG and register test data."""
        if isinstance(testkit, LinkedSourcesTestkit):
            # Register the linked testkit
            linked_key = f"linked_{'_'.join(sorted(testkit.sources.keys()))}"
            self.linked[linked_key] = testkit

            # Add all sources to the real DAG
            for source_testkit in testkit.sources.values():
                self.dag._add_step(source_testkit.source)
                self.sources[source_testkit.name] = source_testkit
        else:
            # Add single source to real DAG
            self.dag._add_step(testkit.source)
            self.sources[testkit.name] = testkit

    def add_model(self, testkit: ModelTestkit) -> None:
        """Add model to the real DAG and register test data."""
        self.dag._add_step(testkit.model)
        self.models[testkit.name] = testkit

    def get_linked_testkit(
        self, name: UnqualifiedResolutionName
    ) -> LinkedSourcesTestkit | None:
        """For a resolution, get the LinkedSourcesTestkit that produced its sources."""
        # Check if it's a source directly
        for linked_testkit in self.linked.values():
            if name in linked_testkit.sources:
                return linked_testkit

        # Check if it's a model - get sources from its queries
        if name in self.models:
            model = self.models[name]

            # Get all sources from left and right queries
            all_sources = list(model.model.left_query.sources)
            if model.model.right_query:
                all_sources.extend(model.model.right_query.sources)

            # Find the linked testkit for the first source we find
            for source in all_sources:
                for linked_testkit in self.linked.values():
                    if source.name in linked_testkit.sources:
                        return linked_testkit

        return None
