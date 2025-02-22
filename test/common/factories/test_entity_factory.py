import pyarrow as pa
import pytest
from faker import Faker

from matchbox.common.factories.entities import (
    EntityReference,
    FeatureConfig,
    ResultsEntity,
    SourceEntity,
    diff_results,
    generate_entities,
    generate_entity_probabilities,
)
from matchbox.common.transform import DisjointSet


@pytest.mark.parametrize(
    ("name", "pks"),
    (
        ("dataset1", frozenset({"1", "2", "3"})),
        ("dataset2", frozenset({"A", "B"})),
    ),
)
def test_entity_reference_creation(name: str, pks: frozenset[str]):
    """Test basic EntityReference creation and access."""
    ref = EntityReference(mapping=frozenset([(name, pks)]))
    assert ref[name] == pks
    assert name in ref
    assert ref["nonexistent"] is None


def test_entity_reference_addition():
    """Test combining EntityReferences."""
    ref1 = EntityReference(mapping=frozenset([("dataset1", frozenset({"1", "2"}))]))
    ref2 = EntityReference(
        mapping=frozenset(
            [("dataset1", frozenset({"2", "3"})), ("dataset2", frozenset({"A"}))]
        )
    )

    combined = ref1 + ref2
    assert combined["dataset1"] == frozenset({"1", "2", "3"})
    assert combined["dataset2"] == frozenset({"A"})


def test_entity_reference_subset():
    """Test subset relationships between EntityReferences."""
    subset = EntityReference(mapping=frozenset([("dataset1", frozenset({"1", "2"}))]))
    superset = EntityReference(
        mapping=frozenset(
            [("dataset1", frozenset({"1", "2", "3"})), ("dataset2", frozenset({"A"}))]
        )
    )

    assert subset <= superset
    assert not superset <= subset


def test_results_entity_creation():
    """Test basic ResultsEntity functionality."""
    ref = EntityReference(mapping=frozenset([("dataset1", frozenset({"1", "2"}))]))
    entity = ResultsEntity(source_pks=ref)

    assert entity.source_pks == ref
    assert isinstance(entity.id, int)


def test_results_entity_addition():
    """Test combining ResultsEntities."""
    entity1 = ResultsEntity(
        source_pks=EntityReference(mapping=frozenset([("dataset1", frozenset({"1"}))]))
    )
    entity2 = ResultsEntity(
        source_pks=EntityReference(mapping=frozenset([("dataset1", frozenset({"2"}))]))
    )

    combined = entity1 + entity2
    assert combined.source_pks["dataset1"] == frozenset({"1", "2"})


def test_source_entity_creation():
    """Test basic SourceEntity functionality."""
    base_values = {"name": "John", "age": 30}
    ref = EntityReference(mapping=frozenset([("dataset1", frozenset({"1", "2"}))]))

    entity = SourceEntity(base_values=base_values, source_pks=ref)

    assert entity.base_values == base_values
    assert entity.source_pks == ref
    assert isinstance(entity.id, int)


@pytest.mark.parametrize(
    ("features", "n"),
    (
        ((FeatureConfig(name="name", base_generator="name"),), 1),
        (
            (
                FeatureConfig(name="name", base_generator="name"),
                FeatureConfig(name="email", base_generator="email"),
            ),
            5,
        ),
    ),
)
def test_generate_entities(features: tuple[FeatureConfig, ...], n: int):
    """Test entity generation with different features and counts."""
    faker = Faker(seed=42)
    entities = generate_entities(faker, features, n)

    assert len(entities) == n
    for entity in entities:
        # Check all features are present
        assert all(f.name in entity.base_values for f in features)
        # Check all values are strings (given our test features)
        assert all(isinstance(v, str) for v in entity.base_values.values())


def test_generate_entity_probabilities():
    """Test probability generation for entity matching."""
    faker = Faker(seed=42)

    # Create some test entities
    source = SourceEntity(
        base_values={"name": "John"},
        source_pks=EntityReference(
            mapping=frozenset([("dataset1", frozenset({"1", "2"}))])
        ),
    )

    results1 = ResultsEntity(
        source_pks=EntityReference(mapping=frozenset([("dataset1", frozenset({"1"}))]))
    )
    results2 = ResultsEntity(
        source_pks=EntityReference(mapping=frozenset([("dataset1", frozenset({"2"}))]))
    )

    # Generate probabilities
    table = generate_entity_probabilities(
        faker, {results1}, {results2}, {source}, (0.8, 1.0)
    )

    # Check table structure
    assert isinstance(table, pa.Table)
    assert table.schema.names == ["left_id", "right_id", "probability"]

    # Check probability range
    probs = table.column("probability").to_numpy()
    assert all(80 <= p <= 100 for p in probs)


def test_generate_entity_probabilities_empty():
    """Test probability generation with empty entity sets."""
    faker = Faker(seed=42)
    table = generate_entity_probabilities(faker, set(), set(), set(), (0.8, 1.0))
    assert len(table) == 0


def test_generate_entity_probabilities_single():
    """Test probability generation with single entity in both sets."""
    faker = Faker(seed=42)
    source = SourceEntity(
        base_values={"name": "Test"},
        source_pks=EntityReference(mapping=frozenset([("dataset1", frozenset({"1"}))])),
    )
    result = ResultsEntity(
        source_pks=EntityReference(mapping=frozenset([("dataset1", frozenset({"1"}))]))
    )

    # Test with identical left and right
    table = generate_entity_probabilities(
        faker, {result}, {result}, {source}, (0.8, 1.0)
    )
    assert len(table) == 0  # Should not match with itself

    # Test with two different entities
    result2 = ResultsEntity(
        source_pks=EntityReference(mapping=frozenset([("dataset1", frozenset({"1"}))]))
    )
    table = generate_entity_probabilities(
        faker, {result}, {result2}, {source}, (0.8, 1.0)
    )
    assert len(table) == 1  # Should match different entities from same source


@pytest.mark.parametrize(
    ("prob_range", "expected_min", "expected_max"),
    (
        ((0.8, 1.0), 80, 100),
        ((0.5, 0.7), 50, 70),
    ),
)
def test_probability_ranges(
    prob_range: tuple[float, float], expected_min: int, expected_max: int
):
    """Test probability generation with different ranges."""
    faker = Faker(seed=42)

    source = SourceEntity(
        base_values={"name": "Test"},
        source_pks=EntityReference(mapping=frozenset([("dataset1", frozenset({"1"}))])),
    )

    results = ResultsEntity(
        source_pks=EntityReference(mapping=frozenset([("dataset1", frozenset({"1"}))]))
    )

    table = generate_entity_probabilities(faker, {results}, None, {source}, prob_range)

    probs = table.column("probability").to_numpy()
    assert all(expected_min <= p <= expected_max for p in probs)


def test_probability_recovers_source_entities():
    """Test that generated probabilities correctly recover source entities."""
    faker = Faker(seed=42)

    # Create two source entities with overlapping PKs across datasets
    source1 = SourceEntity(
        base_values={"name": "John"},
        source_pks=EntityReference(
            mapping=frozenset(
                [("dataset1", frozenset({"1", "2"})), ("dataset2", frozenset({"A"}))]
            )
        ),
    )

    source2 = SourceEntity(
        base_values={"name": "Jane"},
        source_pks=EntityReference(
            mapping=frozenset(
                [("dataset1", frozenset({"3", "4"})), ("dataset2", frozenset({"B"}))]
            )
        ),
    )

    # Create ResultsEntities that split these sources
    results = {
        ResultsEntity(
            source_pks=EntityReference(
                mapping=frozenset([("dataset1", frozenset({"1"}))])
            )
        ),
        ResultsEntity(
            source_pks=EntityReference(
                mapping=frozenset(
                    [("dataset1", frozenset({"2"})), ("dataset2", frozenset({"A"}))]
                )
            )
        ),
        ResultsEntity(
            source_pks=EntityReference(
                mapping=frozenset([("dataset1", frozenset({"3"}))])
            )
        ),
        ResultsEntity(
            source_pks=EntityReference(
                mapping=frozenset(
                    [("dataset1", frozenset({"4"})), ("dataset2", frozenset({"B"}))]
                )
            )
        ),
    }

    # Generate probabilities
    prob_table = generate_entity_probabilities(
        faker, results, None, {source1, source2}, (0.9, 1.0)
    )

    # Use DisjointSet to cluster based on high probabilities
    ds = DisjointSet[int]()
    for row in prob_table.to_pylist():
        if row["probability"] >= 90:  # High confidence matches
            ds.union(row["left_id"], row["right_id"])

    # Get resulting clusters
    clusters = ds.get_components()

    # Should recover original entities - exactly two clusters
    assert len(clusters) == 2

    # Each cluster should contain the right number of ResultsEntities
    cluster_sizes = sorted(len(cluster) for cluster in clusters)
    assert cluster_sizes == [2, 2]  # Each source split into 2 results


def test_complex_entity_relationships():
    """Test generation with complex relationships across multiple datasets."""
    faker = Faker(seed=42)

    # Create source entity present in three datasets
    source = SourceEntity(
        base_values={"name": "Complex"},
        source_pks=EntityReference(
            mapping=frozenset(
                [
                    ("dataset1", frozenset({"1", "2"})),
                    ("dataset2", frozenset({"A", "B"})),
                    ("dataset3", frozenset({"X"})),
                ]
            )
        ),
    )

    # Create ResultsEntities representing partial views
    results = {
        ResultsEntity(
            source_pks=EntityReference(
                mapping=frozenset(
                    [("dataset1", frozenset({"1"})), ("dataset2", frozenset({"A"}))]
                )
            )
        ),
        ResultsEntity(
            source_pks=EntityReference(
                mapping=frozenset(
                    [("dataset2", frozenset({"B"})), ("dataset3", frozenset({"X"}))]
                )
            )
        ),
        ResultsEntity(
            source_pks=EntityReference(
                mapping=frozenset([("dataset1", frozenset({"2"}))])
            )
        ),
    }

    table = generate_entity_probabilities(faker, results, None, {source}, (0.8, 1.0))

    # Should generate probabilities between all relevant pairs
    expected_pairs = 3  # (1-2, 1-3, 2-3) avoiding duplicates
    assert len(table) == expected_pairs


@pytest.mark.parametrize(
    ("expected", "actual", "verbose", "want_identical", "want_msg"),
    [
        # Identical sets
        pytest.param(
            [
                ResultsEntity(
                    source_pks=EntityReference(
                        mapping=frozenset([("d1", frozenset({"1"}))])
                    )
                )
            ],
            [
                ResultsEntity(
                    source_pks=EntityReference(
                        mapping=frozenset([("d1", frozenset({"1"}))])
                    )
                )
            ],
            False,
            True,
            "",
            id="identical_sets",
        ),
        # Complete mismatch
        pytest.param(
            [
                ResultsEntity(
                    source_pks=EntityReference(
                        mapping=frozenset([("d1", frozenset({"1"}))])
                    )
                )
            ],
            [
                ResultsEntity(
                    source_pks=EntityReference(
                        mapping=frozenset([("d1", frozenset({"2"}))])
                    )
                )
            ],
            False,
            False,
            "Mean similarity ratio: 0.00%",
            id="complete_mismatch",
        ),
        # Partial match
        pytest.param(
            [
                ResultsEntity(
                    source_pks=EntityReference(
                        mapping=frozenset([("d1", frozenset({"1", "2"}))])
                    )
                )
            ],
            [
                ResultsEntity(
                    source_pks=EntityReference(
                        mapping=frozenset([("d1", frozenset({"1", "3"}))])
                    )
                )
            ],
            False,
            False,
            "Mean similarity ratio: 33.33%",
            id="partial_match",
        ),
        # Verbose output
        pytest.param(
            [
                ResultsEntity(
                    source_pks=EntityReference(
                        mapping=frozenset([("d1", frozenset({"1", "2"}))])
                    )
                )
            ],
            [
                ResultsEntity(
                    source_pks=EntityReference(
                        mapping=frozenset([("d1", frozenset({"1", "3"}))])
                    )
                )
            ],
            True,
            False,
            # Don't check exact message content since IDs are random
            None,
            id="verbose_output",
        ),
    ],
)
def test_diff_results(
    expected: list[ResultsEntity],
    actual: list[ResultsEntity],
    verbose: bool,
    want_identical: bool,
    want_msg: str | None,
):
    """Test diff_results function handles various scenarios correctly."""
    got_identical, got_msg = diff_results(expected, actual, verbose)
    assert got_identical == want_identical
    if want_msg is not None:  # Skip message check for verbose mode
        assert got_msg == want_msg


def test_source_to_results_conversion():
    """Test converting source entities to results entities and comparing them."""
    # Create source entity present in multiple datasets
    source = SourceEntity(
        base_values={"name": "Test"},
        source_pks=EntityReference(
            mapping=frozenset(
                [
                    ("dataset1", frozenset({"1", "2"})),
                    ("dataset2", frozenset({"A", "B"})),
                ]
            )
        ),
    )

    # Convert different subsets to results entities
    results1 = source.to_results_entity("dataset1")
    results2 = source.to_results_entity("dataset1", "dataset2")
    results3 = source.to_results_entity("dataset2")

    # Test different comparison scenarios
    identical, msg = diff_results([results1], [results1])
    assert identical
    assert msg == ""

    # Compare partial overlap
    identical, msg = diff_results([results1], [results2])
    assert not identical
    assert "dataset2" in str(results2 - results1)

    # Compare disjoint sets
    identical, msg = diff_results([results1], [results3])
    assert not identical
    assert results1.similarity_ratio(results3) == 0.0

    # Test error case for missing dataset
    with pytest.raises(KeyError):
        source.to_results_entity("nonexistent")
