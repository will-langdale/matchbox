from abc import ABC, abstractmethod
from math import comb
from unittest.mock import Mock, create_autospec
from uuid import uuid4

import pandas as pd
import pyarrow as pa
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, create_engine

from matchbox.common.arrow import SCHEMA_INDEX
from matchbox.common.sources import Source, SourceAddress, SourceColumn


class VariationRule(BaseModel, ABC):
    """Abstract base class for variation rules."""

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

    name: str
    base_generator: str
    parameters: dict = Field(default_factory=dict)
    variations: list[VariationRule] = Field(default_factory=list)


class SourceMetrics(BaseModel):
    """Metrics about the generated data."""

    n_true_entities: int
    n_unique_rows: int
    n_potential_pairs: int

    @classmethod
    def calculate(
        cls, n_true_entities: int, max_variations_per_entity: int
    ) -> "SourceMetrics":
        """Calculate metrics based on entity count and variations."""
        n_unique_rows = 1 + max_variations_per_entity
        n_potential_pairs = comb(n_unique_rows, 2) * n_true_entities

        return cls(
            n_true_entities=n_true_entities,
            n_unique_rows=n_unique_rows,
            n_potential_pairs=n_potential_pairs,
        )


class SourceGeneratedData(BaseModel):
    """Contains the generated data and its properties."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pa.Table
    data_hashes: pa.Table
    metrics: SourceMetrics


class SourceDummy(BaseModel):
    """Complete representation of a generated dummy Source."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Source
    features: list[FeatureConfig]
    data: SourceGeneratedData

    def to_mock(self) -> Mock:
        """Create a mock Source object that mimics this dummy source's behavior."""
        mock_source = create_autospec(self.source)

        mock_source.set_engine.return_value = mock_source
        mock_source.default_columns.return_value = mock_source
        mock_source.hash_data.return_value = self.data.data_hashes

        mock_source.model_dump.side_effect = self.source.model_dump
        mock_source.model_dump_json.side_effect = self.source.model_dump_json

        return mock_source


class SourceDataGenerator:
    """Generates dummy data for a Source."""

    def __init__(self, seed: int = 42):
        self.faker = Faker()
        self.faker.seed_instance(seed)

    def generate_data(
        self, n_true_entities: int, features: list[FeatureConfig], repetition: int
    ) -> SourceGeneratedData:
        """Generate raw data as PyArrow tables."""
        max_variations = max(len(f.variations) for f in features)

        raw_data = {"pk": []}
        for feature in features:
            raw_data[feature.name] = []

        # Generate data entity by entity
        for _ in range(n_true_entities):
            # Generate base values -- the raw row
            base_values = {
                f.name: getattr(self.faker, f.base_generator)(**f.parameters)
                for f in features
            }

            raw_data["pk"].append(str(uuid4()))
            for name, value in base_values.items():
                raw_data[name].append(value)

            # Add variations
            for variation_idx in range(max_variations):
                raw_data["pk"].append(str(uuid4()))
                for feature in features:
                    if variation_idx < len(feature.variations):
                        # Apply variation
                        value = feature.variations[variation_idx].apply(
                            base_values[feature.name]
                        )
                    else:
                        # Use base value for padding
                        value = base_values[feature.name]
                    raw_data[feature.name].append(value)

        # Create DataFrame and apply repetition
        df = pd.DataFrame(raw_data)
        df = pd.concat([df] * repetition, ignore_index=True)

        # Group by all features except pk to get hash groups
        feature_names = [f.name for f in features]
        hash_groups = (
            df.groupby(feature_names, sort=False)["pk"].agg(list).reset_index()
        )

        # Create data_hashes table
        hash_groups["hash"] = [str(uuid4()).encode() for _ in range(len(hash_groups))]
        data_hashes = pa.Table.from_pydict(
            {
                "source_pk": hash_groups["pk"].tolist(),
                "hash": hash_groups["hash"].tolist(),
            },
            schema=SCHEMA_INDEX,
        )

        metrics = SourceMetrics.calculate(
            n_true_entities=n_true_entities, max_variations_per_entity=max_variations
        )

        return SourceGeneratedData(
            data=pa.Table.from_pandas(df),
            data_hashes=data_hashes,
            metrics=metrics,
        )


def source_factory(
    features: list[FeatureConfig] | list[dict] | None = None,
    full_name: str | None = None,
    engine: Engine | None = None,
    n_true_entities: int = 10,
    repetition: int = 1,
    seed: int = 42,
) -> SourceDummy:
    """Generate a complete dummy source.

    Args:
        features: List of FeatureConfigs, used to generate features with variations
        full_name: Full name of the source, like "dbt.companies_house".
        engine: SQLAlchemy engine to use for the source.
        n_true_entities: Number of true entities to generate.
        repetition: Number of times to repeat the data.
        seed: Random seed for data generation.

    Returns:
        SourceDummy: Complete dummy source with generated data.
    """
    generator = SourceDataGenerator(seed)

    if features is None:
        features = [
            FeatureConfig(
                name="company_name",
                base_generator="company",
            ),
            FeatureConfig(
                name="crn",
                base_generator="bothify",
                parameters={"text": "???-###-???-###"},
            ),
        ]

    if full_name is None:
        full_name = generator.faker.word()

    if engine is None:
        engine = create_engine("sqlite:///:memory:")

    if features and isinstance(features[0], dict):
        features = [FeatureConfig.model_validate(feature) for feature in features]

    generated_data = generator.generate_data(
        n_true_entities=n_true_entities, features=features, repetition=repetition
    )

    source = Source(
        address=SourceAddress.compose(full_name=full_name, engine=engine),
        db_pk="pk",
        columns=[
            SourceColumn(name=feature.name, alias=feature.name) for feature in features
        ],
    )

    return SourceDummy(
        source=source,
        features=features,
        data=generated_data,
    )
