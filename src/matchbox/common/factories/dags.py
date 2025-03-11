from pydantic import BaseModel

from matchbox.common.factories.models import ModelTestkit
from matchbox.common.factories.sources import LinkedSourcesTestkit, SourceTestkit


class TestkitDAG(BaseModel):
    """Simple DAG container for testkits."""

    sources: dict[str, SourceTestkit] = {}
    linked: list[LinkedSourcesTestkit] = []
    source_to_linked: dict[str, int] = {}
    models: dict[str, ModelTestkit] = {}

    # Dependency graph tracking
    adjacency: dict[str, set[str]] = {}  # name -> direct dependencies
    root_sources: dict[str, set[str]] = {}  # model -> root sources

    # Keep track of all used names to ensure uniqueness
    _all_names: set[str] = set()

    def _validate_unique_name(self, name: str) -> str:
        """Ensure a name is unique across all testkits."""
        if name in self._all_names:
            raise ValueError(
                f"Name collision detected: '{name}' is already used in this DAG"
            )
        self._all_names.add(name)
        return name

    def _update_root_sources(self, model_name: str):
        """Update root sources for a model."""
        for dep_name in self.adjacency[model_name]:
            if dep_name in self.sources:
                # If dependency is a source, add it directly
                self.root_sources[model_name].add(dep_name)
            elif dep_name in self.models:
                # If dependency is a model, add all its root sources
                self.root_sources[model_name].update(self.root_sources[dep_name])

    def add_source(self, testkit: SourceTestkit | LinkedSourcesTestkit):
        """Add a source testkit to the DAG."""
        if isinstance(testkit, LinkedSourcesTestkit):
            self.linked.append(testkit)
            linked_idx = len(self.linked) - 1
            for source_name, source_testkit in testkit.sources.items():
                self._validate_unique_name(source_name)
                self.sources[source_name] = source_testkit
                self.source_to_linked[source_name] = linked_idx
        else:
            source_name = self._validate_unique_name(testkit.name)
            self.sources[source_name] = testkit

    def add_model(self, testkit: ModelTestkit):
        """Add a model testkit to the DAG."""
        model_name = self._validate_unique_name(testkit.name)
        self.models[model_name] = testkit
        self.adjacency[model_name] = set()
        self.root_sources[model_name] = set()

        # Track dependencies based on left/right resolution
        left_res = testkit.model.metadata.left_resolution
        if left_res:
            self.adjacency[model_name].add(left_res)

        right_res = testkit.model.metadata.right_resolution
        if right_res and right_res != left_res:
            self.adjacency[model_name].add(right_res)

        # Update root sources
        self._update_root_sources(model_name)

    def get_sources_for_model(
        self, model_name: str
    ) -> tuple[LinkedSourcesTestkit | None, list[str]]:
        """Find the LinkedSourcesTestkit and specific source names that a model uses.

        Args:
            model_name: The name of the model to analyze

        Returns:
            A tuple containing:
            - The LinkedSourcesTestkit (or None if not found)
            - A list of source names that the model merges/dedupes
        """
        model = self.models.get(model_name)
        if not model:
            return None, []

        # Get root sources for this model
        root_sources = self.root_sources.get(model_name, set())

        # Find root sources that have a linked testkit
        linked_sources = {s for s in root_sources if s in self.source_to_linked}

        if not linked_sources:
            return None, []

        # Get the first linked source and its testkit
        first_linked_source = next(iter(linked_sources))
        linked_idx = self.source_to_linked[first_linked_source]
        linked_testkit = self.linked[linked_idx]

        # Get all sources from this linked testkit
        linked_source_names = set(linked_testkit.sources.keys())

        # Find which root sources belong to this linked testkit
        model_sources = sorted(list(root_sources.intersection(linked_source_names)))

        return linked_testkit, model_sources
