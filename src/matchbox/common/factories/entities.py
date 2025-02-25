"""Classes and functions for generating and comparing entities.

These underpin the entity resolution process, which is the core of the
dummy sources and models factory system.
"""

import datetime
from abc import ABC, abstractmethod
from decimal import Decimal
from functools import cache
from random import getrandbits
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from faker import Faker
from frozendict import frozendict
from pydantic import BaseModel, ConfigDict, Field, field_validator

from matchbox.common.transform import DisjointSet

if TYPE_CHECKING:
    from matchbox.common.factories.sources import SourceDummy
else:
    SourceDummy = Any


class VariationRule(BaseModel, ABC):
    """Abstract base class for variation rules."""

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def apply(self, value: str) -> str:
        """Apply the variation to a value."""
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """Return the type of variation."""
        pass


class SuffixRule(VariationRule):
    """Add a suffix to a value."""

    suffix: str

    def apply(self, value: str) -> str:
        return f"{value}{self.suffix}"

    @property
    def type(self) -> str:
        return "suffix"


class PrefixRule(VariationRule):
    """Add a prefix to a value."""

    prefix: str

    def apply(self, value: str) -> str:
        return f"{self.prefix}{value}"

    @property
    def type(self) -> str:
        return "prefix"


class ReplaceRule(VariationRule):
    """Replace occurrences of a string with another."""

    old: str
    new: str

    def apply(self, value: str) -> str:
        return value.replace(self.old, self.new)

    @property
    def type(self) -> str:
        return "replace"


def infer_sql_type(base: str, parameters: tuple) -> str:
    """
    Infer an appropriate SQL type from a Faker configuration.

    Args:
        base: Faker generator type
        parameters: Parameters for the generator

    Returns:
        A SQL type string
    """
    """Generate base entities with their ground truth values."""
    generator = Faker()
    value_generator = getattr(generator, base)
    examples = [value_generator(**dict(parameters)) for _ in range(5)]

    # Get the types of all non-None examples
    types_found = {type(x) for x in examples}

    # If multiple types, use the most general one
    if len(types_found) > 1:
        # Check for numeric types
        if all(issubclass(t, (int, float, Decimal)) for t in types_found):
            if any(issubclass(t, float) or issubclass(t, Decimal) for t in types_found):
                return "FLOAT"
            return "INTEGER"
        # Default to TEXT for mixed types
        return "TEXT"

    # Single type case
    python_type = next(iter(types_found))

    type_map = {
        str: "TEXT",
        int: "INTEGER",
        float: "FLOAT",
        bool: "BOOLEAN",
        datetime.datetime: "TIMESTAMP",
        datetime.date: "DATE",
        datetime.time: "TIME",
        Decimal: "DECIMAL(10,2)",
        bytes: "BLOB",
        bytearray: "BLOB",
    }

    return type_map.get(python_type, "TEXT")


class FeatureConfig(BaseModel):
    """Configuration for generating a feature with variations."""

    model_config = ConfigDict(frozen=True)

    name: str
    base_generator: str
    parameters: tuple = Field(
        default_factory=tuple,
        description=(
            "Parameters for the generator. A tuple of tuples passed to the generator."
        ),
    )
    unique: bool = Field(
        default=True,
        description=(
            "Whether the generator enforces uniqueness in the generated data. "
            "For example, using unique=True with the 'boolean' generator will error "
            "if more the two values are generated."
        ),
    )
    drop_base: bool = Field(
        default=False, description="Whether the base case is dropped."
    )
    variations: tuple[VariationRule, ...] = Field(default_factory=tuple)
    sql_type: str = Field(
        default_factory=lambda data: infer_sql_type(
            data["base_generator"], data["parameters"]
        )
    )

    def add_variations(self, *rule: VariationRule) -> "FeatureConfig":
        """Add a variation rule to the feature."""
        return FeatureConfig(
            name=self.name,
            base_generator=self.base_generator,
            parameters=self.parameters,
            unique=self.unique,
            drop_base=self.drop_base,
            variations=self.variations + tuple(rule),
        )

    @field_validator("name", mode="after")
    def protected_names(cls, value: str) -> str:
        """Ensure name is not a reserved keyword."""
        if value in {"id", "pk"}:
            raise ValueError("Feature name cannot be 'id' or 'pk'.")
        return value


class EntityReference(frozendict):
    """Reference to an entity's presence in specific sources.

    Maps dataset names to sets of primary keys.
    """

    def __init__(self, mapping: dict[str, frozenset[str]] | None = None) -> None:
        super().__init__({} if mapping is None else mapping)

    def __add__(self, other: "EntityReference") -> "EntityReference":
        """Merge two EntityReferences by unioning PKs for each dataset"""
        if not isinstance(other, EntityReference):
            return NotImplemented

        return EntityReference(
            {
                k: self.get(k, frozenset()) | other.get(k, frozenset())
                for k in self.keys() | other.keys()
            }
        )

    def __le__(self, other: "EntityReference") -> bool:
        """Test if self is a subset of other"""
        if not isinstance(other, EntityReference):
            return NotImplemented

        return all(name in other and self[name] <= other[name] for name in self)


class EntityIDMixin:
    """Mixin providing common ID-based functionality for entity classes.

    Implements integer conversion and comparison operators for sorting
    based on the entity's ID.
    """

    id: int

    def __int__(self) -> int:
        """Allow converting an entity to an integer by returning its ID."""
        return self.id

    def __lt__(self, other: int | Any) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id < other.id
        if isinstance(other, int):
            return self.id < other
        return NotImplemented

    def __gt__(self, other: int | Any) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id > other.id
        if isinstance(other, int):
            return self.id > other
        return NotImplemented

    def __le__(self, other: int | Any) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id <= other.id
        if isinstance(other, int):
            return self.id <= other
        return NotImplemented

    def __ge__(self, other: int | Any) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id >= other.id
        if isinstance(other, int):
            return self.id >= other
        return NotImplemented


class SourcePKMixin:
    """Mixin providing common source primary key functionality for entity classes.

    Implements methods for accessing and retrieving source primary keys.
    """

    source_pks: EntityReference

    def get_source_pks(self, source_name: str) -> set[str]:
        """Get PKs for a specific source.

        Args:
            source_name: Name of the dataset

        Returns:
            Set of primary keys, empty if dataset not found
        """
        return set(self.source_pks.get(source_name, frozenset()))

    def get_values(
        self, sources: dict[str, "SourceDummy"]
    ) -> dict[str, dict[str, list[str]]]:
        """Get all unique values for this entity across sources.

        Each source may have its own variations/transformations of the base data,
        so we maintain separation between sources.

        Args:
            sources: Dictionary of source name to source data

        Returns:
            Dictionary mapping:
                source_name -> {
                    feature_name -> [unique values for that feature in that source]
                }
        """
        values: dict[str, dict[str, list[str]]] = {}

        # For each dataset we have PKs for
        for dataset_name, pks in self.source_pks.items():
            source = sources.get(dataset_name)

            if source is None:
                raise ValueError(f"Source not found: {dataset_name}")

            # Get rows for this entity in this source
            df = source.data.to_pandas()
            entity_rows = df[df["pk"].isin(pks)]

            # Get unique values for each feature in this source
            values[dataset_name] = {
                feature.name: sorted(entity_rows[feature.name].unique())
                for feature in source.features
            }

        return values


class ResultsEntity(BaseModel, EntityIDMixin, SourcePKMixin):
    """Represents a merged entity mid-pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id: int = Field(default_factory=lambda: getrandbits(63))
    source_pks: EntityReference

    def __add__(self, other: "ResultsEntity") -> "ResultsEntity":
        """Combine two ResultsEntities by combining the source_pks."""
        if other is None:
            return self
        if not isinstance(other, ResultsEntity):
            return NotImplemented
        return ResultsEntity(source_pks=self.source_pks + other.source_pks)

    def __radd__(self, other: Any) -> "ResultsEntity":
        """Handle sum() by treating 0 as an empty ResultsEntity."""
        if other == 0:  # sum() starts with 0
            return self
        return NotImplemented

    def __sub__(self, other: "ResultsEntity") -> dict[str, frozenset[str]]:
        """Return PKs in self that aren't in other, by dataset.

        Used to diff two ResultsEntities.
        """
        if not isinstance(other, ResultsEntity):
            return NotImplemented

        diff = {}
        for name, our_pks in self.source_pks.items():
            their_pks = other.source_pks.get(name, frozenset())
            if remaining := our_pks - their_pks:
                diff[name] = remaining

        return diff

    def __rsub__(self, other: "ResultsEntity") -> dict[str, frozenset[str]]:
        """Support reverse subtraction."""
        if not isinstance(other, ResultsEntity):
            return NotImplemented
        return other - self

    def __eq__(self, other: Any) -> bool:
        """Compare based on source_pks."""
        if not isinstance(other, ResultsEntity):
            return NotImplemented
        return self.source_pks == other.source_pks

    def __contains__(self, other: "ResultsEntity") -> bool:
        """Check if this entity contains all PKs from other entity."""
        return other.source_pks <= self.source_pks

    def __hash__(self) -> int:
        """Hash based on EntityReference which is itself hashable."""
        return hash(self.source_pks)

    def is_subset_of_source_entity(self, source_entity: "SourceEntity") -> bool:
        """Check if this ResultsEntity's references are a subset of a SourceEntity's."""
        return self.source_pks <= source_entity.source_pks

    def similarity_ratio(self, other: "ResultsEntity") -> float:
        """Return ratio of shared PKs to total PKs across all datasets."""
        total_pks = 0
        shared_pks = 0

        # Get all dataset names
        all_datasets = set(self.source_pks.keys()) | set(other.source_pks.keys())

        for name in all_datasets:
            our_pks = self.source_pks.get(name, frozenset())
            their_pks = other.source_pks.get(name, frozenset())

            total_pks += len(our_pks | their_pks)
            shared_pks += len(our_pks & their_pks)

        return shared_pks / total_pks if total_pks > 0 else 0.0


class SourceEntity(BaseModel, EntityIDMixin, SourcePKMixin):
    """Represents a single entity across all sources."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(default_factory=lambda: getrandbits(63))
    base_values: dict[str, Any] = Field(description="Feature name -> base value")
    source_pks: EntityReference = Field(
        description="Dataset to PKs mapping",
        default=EntityReference(mapping=frozenset()),
    )
    total_unique_variations: int = Field(default=0)

    def __eq__(self, other: object) -> bool:
        """Equal if base values are shared, or integer ID matches."""
        if isinstance(other, SourceEntity):
            return self.base_values == other.base_values
        if isinstance(other, int):
            return self.id == other
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on sorted base values."""
        return hash(tuple(sorted(self.base_values.items())))

    def add_source_reference(self, name: str, pks: list[str]) -> None:
        """Add or update a source reference.

        Args:
            name: Dataset name
            pks: List of primary keys for this dataset
        """
        mapping = dict(self.source_pks)
        mapping[name] = frozenset(pks)
        self.source_pks = EntityReference(mapping)

    def to_results_entity(self, *names: str) -> ResultsEntity:
        """Convert this SourceEntity to a ResultsEntity with the specified datasets.

        This method makes diffing really easy. Testing whether ResultEntity objects
        are subsets of SourceEntity objects is a weaker, logically more fragile test
        than directly comparing equality of sets of ResultEntity objects. It enables
        a really simple syntactical expression of the test.

        ```python
        actual: set[ResultsEntity] = ...
        expected: set[ResultsEntity] = {
            s.to_results_entity("dataset1", "dataset2")
            for s in source_entities
        }

        is_identical = expected) == actual
        missing = expected - actual
        extra = actual - expected
        ```

        Args:
            *names: Names of datasets to include in the ResultsEntity

        Returns:
            ResultsEntity containing only the specified datasets' PKs

        Raises:
            KeyError: If any specified dataset name doesn't exist in this entity
        """
        missing_datasets = set(names) - self.source_pks.keys()
        if missing_datasets:
            raise KeyError(f"Datasets not found in entity: {missing_datasets}")

        filtered = {name: self.source_pks[name] for name in names}
        return ResultsEntity(source_pks=EntityReference(filtered))


@cache
def generate_entities(
    generator: Faker,
    features: tuple[FeatureConfig, ...],
    n: int,
) -> tuple[SourceEntity]:
    """Generate base entities with their ground truth values."""
    entities = []
    for _ in range(n):
        base_values = {}
        for feature in features:
            generator_func = generator.unique if feature.unique else generator
            value_generator = getattr(generator_func, feature.base_generator)
            base_values[feature.name] = value_generator(**dict(feature.parameters))

        entities.append(
            SourceEntity(base_values=base_values, source_pks=EntityReference())
        )
    return tuple(entities)


def probabilities_to_results_entities(
    probabilities: pa.Table,
    left_results: tuple[ResultsEntity, ...],
    right_results: tuple[ResultsEntity, ...] | None = None,
    threshold: float | int = 0,
) -> tuple[ResultsEntity, ...]:
    """Convert probabilities to ResultsEntities based on a threshold."""
    left_lookup = {entity.id: entity for entity in left_results}
    if right_results is not None:
        right_lookup = {entity.id: entity for entity in right_results}
    else:
        right_lookup = left_lookup

    djs = DisjointSet[ResultsEntity]()

    # Validate threshold
    if isinstance(threshold, float):
        threshold = int(threshold * 100)

    # Add ALL entities to the disjoint set
    for entity in left_results:
        djs.add(entity)
    if right_results is not None:
        for entity in right_results:
            djs.add(entity)

    # Add edges to the disjoint set
    for record in probabilities.to_pylist():
        if record["probability"] >= threshold:
            djs.union(
                left_lookup.get(record["left_id"]),
                right_lookup.get(record["right_id"]),
            )

    components: set[set[ResultsEntity]] = djs.get_components()

    entities: list[ResultsEntity] = []
    for component in components:
        merged: ResultsEntity = sum(component)
        entities.append(merged)

    return tuple(entities)


def diff_results(
    expected: list[ResultsEntity], actual: list[ResultsEntity], verbose: bool = False
) -> tuple[bool, dict]:
    """Compare two lists of ResultsEntity with detailed diff information.

    Args:
        expected: Expected ResultsEntity list
        actual: Actual ResultsEntity list
        verbose: Whether to return detailed diff report

    Returns:
        Tuple of (is_identical, diff_dict)
    """
    expected_set = set(expected)
    actual_set = set(actual)

    if expected_set == actual_set:
        return True, {}

    missing = expected_set - actual_set
    extra = actual_set - expected_set

    # Calculate similarity scores and determine partial matches
    partial_matches = {}
    best_match_entities = set()

    # Calculate mean similarity (best matches for all entities)
    best_ratios = []

    # Best ratios for missing entities
    for m in missing:
        # Find best match for this missing entity
        best_match = None
        best_ratio = 0.0

        for a in actual:
            ratio = m.similarity_ratio(a)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = a

        best_ratios.append(best_ratio)

        # Record partial match if similarity > 0
        if best_ratio > 0:
            best_match_entities.add(best_match)

            # Calculate difference between entities
            missing_pks = m - best_match
            extra_pks = best_match - m

            partial_matches[m] = {
                "entity": best_match,
                "similarity": best_ratio,
                "missing_pks": missing_pks,
                "extra_pks": extra_pks,
            }

    # Best ratios for extra entities (including those used as best matches)
    for e in extra:
        best_ratio = max((e.similarity_ratio(m) for m in expected), default=0.0)
        best_ratios.append(best_ratio)

    # Calculate mean similarity
    mean_ratio = sum(best_ratios) / len(best_ratios) if best_ratios else 0.0

    # Build basic result dictionary
    result = {"mean_similarity": mean_ratio, "partial": [], "missing": [], "extra": []}

    if not verbose:
        return False, result

    # Build detailed result dictionary when verbose=True
    # Add partial matches
    for m, match_info in partial_matches.items():
        result["partial"].append(
            {
                "missing_entity_id": m.id,
                "matches": [
                    {
                        "actual_entity_id": match_info["entity"].id,
                        "similarity": match_info["similarity"],
                        "missing_pks": dict(match_info["missing_pks"])
                        if match_info["missing_pks"]
                        else {},
                        "extra_pks": dict(match_info["extra_pks"])
                        if match_info["extra_pks"]
                        else {},
                    }
                ],
            }
        )

    # Add completely missing entities (those with no partial matches)
    completely_missing = missing - set(partial_matches.keys())
    for e in completely_missing:
        result["missing"].append({"id": e.id, "source_pks": dict(e.source_pks.items())})

    # Add extra entities (excluding those used in partial matches)
    for e in extra - best_match_entities:
        result["extra"].append({"id": e.id, "source_pks": dict(e.source_pks.items())})

    return False, result
