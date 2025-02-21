from abc import ABC, abstractmethod
from functools import cache
from random import getrandbits
from typing import TYPE_CHECKING, Any, Iterator

import pyarrow as pa
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class FeatureConfig(BaseModel):
    """Configuration for generating a feature with variations."""

    model_config = ConfigDict(frozen=True)

    name: str
    base_generator: str
    parameters: tuple = Field(default_factory=tuple)
    unique: bool = Field(default=True)
    drop_base: bool = Field(default=False)
    variations: tuple[VariationRule, ...] = Field(default_factory=tuple)

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


class EntityReference(BaseModel):
    """Reference to an entity's presence in a specific source.

    Implemented as a frozenset rather than a dict for speed and immutability.
    """

    mapping: frozenset[tuple[str, frozenset[str]]] = Field(
        description="(dataset_name, pks) pairs"
    )

    model_config = ConfigDict(frozen=True)

    def __getitem__(self, dataset: str) -> frozenset[str] | None:
        """Get PKs for a dataset if it exists"""
        for name, pks in self.mapping:
            if name == dataset:
                return pks
        return None

    def items(self) -> Iterator[tuple[str, frozenset[str]]]:
        """Iterator over (dataset, pks) pairs"""
        return iter(self.mapping)

    def __contains__(self, dataset: str) -> bool:
        """Check if dataset exists in mapping"""
        return any(name == dataset for name, _ in self.mapping)

    def __hash__(self) -> int:
        """Hash based on the immutable mapping"""
        return hash(self.mapping)

    def __eq__(self, other: Any) -> bool:
        """Equal if mappings are identical"""
        if not isinstance(other, EntityReference):
            return NotImplemented
        return self.mapping == other.mapping

    def __add__(self, other: "EntityReference") -> "EntityReference":
        """Merge two EntityReferences by unioning PKs for each dataset"""
        if not isinstance(other, EntityReference):
            return NotImplemented

        # Build combined mapping
        combined = {}
        # Add all our datasets
        for name, pks in self.mapping:
            combined[name] = pks

        # Merge in other's datasets
        for name, pks in other.mapping:
            if name in combined:
                # Union PKs for shared datasets
                combined[name] = combined[name] | pks
            else:
                combined[name] = pks

        # Convert back to frozenset of tuples
        new_mapping = frozenset((name, pks) for name, pks in combined.items())
        return EntityReference(mapping=new_mapping)

    def __le__(self, other: "EntityReference") -> bool:
        """Test if self is a subset of other"""
        if not isinstance(other, EntityReference):
            return NotImplemented

        # For each of our datasets
        for name, our_pks in self.mapping:
            # Get their PKs for this dataset
            their_pks = other[name]
            # If they don't have this dataset or our PKs aren't a subset,
            # we're not a subset
            if their_pks is None or not our_pks <= their_pks:
                return False
        return True


class ResultsEntity(BaseModel):
    """Represents a merged entity mid-pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id: int = Field(default_factory=lambda: getrandbits(63))
    source_pks: EntityReference

    def __add__(self, other: "ResultsEntity") -> "ResultsEntity":
        """Combine two ResultsEntities by combining the source_pks."""
        if not isinstance(other, ResultsEntity):
            return NotImplemented
        return ResultsEntity(source_pks=self.source_pks + other.source_pks)

    def __eq__(self, other: Any) -> bool:
        """Compare based on source_pks."""
        if not isinstance(other, ResultsEntity):
            return NotImplemented
        return self.source_pks == other.source_pks

    def __lt__(self, other: Any) -> bool:
        """Compare based on ID for sorting operations."""
        if isinstance(other, ResultsEntity):
            return self.id < other.id
        if isinstance(other, int):
            return self.id < other
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        """Compare based on ID for sorting operations."""
        if isinstance(other, ResultsEntity):
            return self.id > other.id
        if isinstance(other, int):
            return self.id > other
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        """Compare based on ID for sorting operations."""
        if isinstance(other, ResultsEntity):
            return self.id <= other.id
        if isinstance(other, int):
            return self.id <= other
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        """Compare based on ID for sorting operations."""
        if isinstance(other, ResultsEntity):
            return self.id >= other.id
        if isinstance(other, int):
            return self.id >= other
        return NotImplemented

    def __hash__(self) -> int:
        """Hash now directly uses the frozenset hash."""
        return hash(self.source_pks)

    def __int__(self) -> int:
        return self.id

    def is_subset_of(self, source_entity: "SourceEntity") -> bool:
        """Check if this ResultsEntity's references are a subset of a SourceEntity's."""
        return self.source_pks <= source_entity.source_pks


class SourceEntity(BaseModel):
    """Represents a single entity across all sources."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(default_factory=lambda: getrandbits(63))
    base_values: dict[str, Any] = Field(description="Feature name -> base value")
    source_pks: EntityReference = Field(
        description="Dataset to PKs mapping", default=frozenset()
    )
    total_unique_variations: int = Field(default=0)

    def __eq__(self, other: object) -> bool:
        """Equal if base values are shared, or integer ID matches."""
        if isinstance(other, SourceEntity):
            return self.base_values == other.base_values
        if isinstance(other, int):
            return self.id == other
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Compare based on ID for sorting operations."""
        if isinstance(other, (SourceEntity, int)):
            return self.id < int(other)
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        """Compare based on ID for sorting operations."""
        if isinstance(other, (SourceEntity, int)):
            return self.id > int(other)
        return NotImplemented

    def __le__(self, other: object) -> bool:
        """Compare based on ID for sorting operations."""
        if isinstance(other, (SourceEntity, int)):
            return self.id <= int(other)
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        """Compare based on ID for sorting operations."""
        if isinstance(other, (SourceEntity, int)):
            return self.id >= int(other)
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on sorted base values."""
        return hash(tuple(sorted(self.base_values.items())))

    def __int__(self) -> int:
        return self.id

    def add_source_reference(self, name: str, pks: list[str]) -> None:
        """Add or update a source reference.

        Args:
            name: Dataset name
            pks: List of primary keys for this dataset
        """
        # Get existing mapping pairs
        pairs = {name: pks for name, pks in self.source_pks.mapping}
        # Update or add new dataset
        pairs[name] = frozenset(pks)
        # Create new EntityReference with updated mapping
        self.source_pks = EntityReference(
            mapping=frozenset((name, pks) for name, pks in pairs.items())
        )

    def get_source_pks(self, source_name: str) -> set[str]:
        """Get PKs for a specific source.

        Args:
            source_name: Name of the dataset

        Returns:
            List of primary keys, empty if dataset not found
        """
        pks = self.source_pks[source_name]
        return set(pks) if pks is not None else []

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
                continue

            # Get rows for this entity in this source
            df = source.data.to_pandas()
            entity_rows = df[df["pk"].isin(pks)]

            # Get unique values for each feature in this source
            values[dataset_name] = {
                feature.name: sorted(entity_rows[feature.name].unique())
                for feature in source.features
            }

        return values


@cache
def generate_entities(
    generator: Faker,
    features: tuple[FeatureConfig, ...],
    n: int,
) -> tuple[SourceEntity]:
    """Generate base entities with their ground truth values."""
    entities = []
    for _ in range(n):
        base_values = {
            f.name: getattr(
                generator.unique if f.unique else generator, f.base_generator
            )(**dict(f.parameters))
            for f in features
        }
        entities.append(
            SourceEntity(
                base_values=base_values, source_pks=EntityReference(mapping=frozenset())
            )
        )
    return tuple(entities)


# @cache
def generate_entity_probabilities(
    generator: Faker,
    left_entities: set[ResultsEntity],
    right_entities: set[ResultsEntity] | None,
    source_entities: set[SourceEntity],
    prob_range: tuple[float, float] = (0.8, 1.0),
) -> pa.Table:
    """Generate probabilities that will recover entity relationships."""
    if right_entities is None:
        right_entities = left_entities

    prob_min = int(prob_range[0] * 100)
    prob_max = int(prob_range[1] * 100)

    left_ids = []
    right_ids = []

    # For each left entity
    for left_entity in left_entities:
        # Find its source entity
        left_source = None
        for source in source_entities:
            if left_entity.is_subset_of(source):
                left_source = source
                break

        if left_source is None:
            continue

        # For each right entity
        for right_entity in right_entities:
            # Skip exact same entity
            if left_entity is right_entity:
                continue

            # Skip self-matches in deduplication case (>= is arbitrary)
            if right_entities is left_entities and left_entity.id >= right_entity.id:
                continue

            # Check if maps to same source
            if any(
                right_entity.is_subset_of(source) and source == left_source
                for source in source_entities
            ):
                left_ids.append(left_entity.id)
                right_ids.append(right_entity.id)

    # Generate probabilities
    probabilities = [
        generator.random_int(min=prob_min, max=prob_max) for _ in range(len(left_ids))
    ]

    return pa.Table.from_arrays(
        [
            pa.array(left_ids, type=pa.uint64()),
            pa.array(right_ids, type=pa.uint64()),
            pa.array(probabilities, type=pa.uint8()),
        ],
        names=["left_id", "right_id", "probability"],
    )


if __name__ == "__main__":
    pass
