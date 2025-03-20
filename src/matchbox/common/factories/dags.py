"""DAG container for testkits."""

from pydantic import BaseModel

from matchbox.common.factories.models import ModelTestkit
from matchbox.common.factories.sources import LinkedSourcesTestkit, SourceTestkit


class TestkitDAG(BaseModel):
    """Simple DAG container for testkits."""

    sources: dict[str, SourceTestkit] = {}
    linked: dict[str, LinkedSourcesTestkit] = {}
    source_to_linked: dict[str, str] = {}
    models: dict[str, ModelTestkit] = {}

    # Dependency graph tracking
    adjacency: dict[str, set[str]] = {}  # name -> direct dependencies
    root_source_addresses: dict[str, set[str]] = {}  # model -> root source addresses

    # Keep track of all used names to ensure uniqueness
    _all_names: set[str] = set()

    @property
    def source_address_to_name(self) -> dict[str, str]:
        """Map source address string to source key."""
        return {str(tk.source.address): name for name, tk in self.sources.items()}

    def _validate_unique_name(self, name: str) -> str:
        """Ensure a name is unique across all testkits."""
        if name in self._all_names:
            raise ValueError(
                f"Name collision detected: '{name}' is already used in this DAG"
            )
        self._all_names.add(name)
        return name

    def _validate_dependencies(self, left_res: str, right_res: str | None) -> bool:
        """Ensure dependencies are valid."""
        valid_deps = set(self.source_address_to_name.keys()) | set(self.models.keys())
        dependencies = {dep for dep in [left_res, right_res] if dep is not None}

        missing_deps = dependencies - valid_deps
        if missing_deps:
            raise ValueError(
                f"Missing dependencies for model: {missing_deps}. "
                "Ensure all dependencies are added before models."
            )
        return True

    def _update_root_source_addresses(self, model_name: str):
        """Update root source addresses for a model."""
        for dep_name in self.adjacency[model_name]:
            if dep_name in self.source_address_to_name:
                # If dependency is a source, add it directly
                self.root_source_addresses[model_name].add(dep_name)
            elif dep_name in self.models:
                # If dependency is a model, add all its root sources
                self.root_source_addresses[model_name].update(
                    self.root_source_addresses[dep_name]
                )

    def add_source(self, testkit: SourceTestkit | LinkedSourcesTestkit):
        """Add a source testkit to the DAG."""
        if isinstance(testkit, LinkedSourcesTestkit):
            # Create linked key as "linked_" + concatenated source names
            source_names = sorted(testkit.sources.keys())
            linked_key = f"linked_{'_'.join(source_names)}"

            # Store LinkedSourcesTestkit in the linked dict with the new key
            self.linked[linked_key] = testkit

            # Add each source and track its association with the linked key
            for source_name, source_testkit in testkit.sources.items():
                self._validate_unique_name(source_name)
                self.sources[source_name] = source_testkit
                self.source_to_linked[source_name] = linked_key
        else:
            source_name = self._validate_unique_name(testkit.name)
            self.sources[source_name] = testkit

    def add_model(self, testkit: ModelTestkit):
        """Add a model testkit to the DAG."""
        model_name = self._validate_unique_name(testkit.name)
        self.models[model_name] = testkit
        self.adjacency[model_name] = set()
        self.root_source_addresses[model_name] = set()

        # Validate dependencies
        left_res = testkit.model.metadata.left_resolution
        right_res = testkit.model.metadata.right_resolution
        self._validate_dependencies(left_res, right_res)

        # Track dependencies based on left/right resolution
        if left_res:
            self.adjacency[model_name].add(left_res)

        if right_res and right_res != left_res:
            self.adjacency[model_name].add(right_res)

        # Update root sources
        self._update_root_source_addresses(model_name)

    def get_sources_for_model(self, model_name: str) -> dict[str | None, set[str]]:
        """Find the LinkedSourcesTestkit keys and specific source names for a model.

        Args:
            model_name: The name of the model to analyze

        Returns:
            A dictionary mapping:
            - LinkedSourcesTestkit keys (or None) to sets of model's source names
        """
        if model_name not in self.models:
            return {None: set()}

        root_source_addresses = self.root_source_addresses.get(model_name, set())
        if not root_source_addresses:
            return {None: set()}

        source_names = [
            self.source_address_to_name[address] for address in root_source_addresses
        ]

        result = {}
        for name in source_names:
            # linked_key could be None
            linked_key = self.source_to_linked.get(name)
            if linked_key not in result:
                result[linked_key] = set()
            result[linked_key].add(name)

        return result
