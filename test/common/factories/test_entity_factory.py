from typing import Any

import pyarrow as pa
import pytest
from faker import Faker

from matchbox.common.factories.entities import (
    ClusterEntity,
    EntityReference,
    FeatureConfig,
    SourceEntity,
    diff_results,
    generate_entities,
    probabilities_to_results_entities,
)


def make_cluster_entity(id: int, dataset: str, pks: list[str]) -> ClusterEntity:
    """Helper to create a ClusterEntity with specified dataset and PKs."""
    return ClusterEntity(id=id, source_pks=EntityReference({dataset: frozenset(pks)}))


def make_source_entity(dataset: str, pks: list[str], base_val: str) -> SourceEntity:
    """Helper to create a SourceEntity with specified dataset and PKs."""
    entity = SourceEntity(base_values={"name": base_val})
    entity.add_source_reference(dataset, pks)
    return entity


@pytest.mark.parametrize(
    ("name", "pks"),
    (
        ("dataset1", frozenset({"1", "2", "3"})),
        ("dataset2", frozenset({"A", "B"})),
    ),
)
def test_entity_reference_creation(name: str, pks: frozenset[str]):
    """Test basic EntityReference creation and access."""
    ref = EntityReference({name: pks})
    assert ref[name] == pks
    assert name in ref
    with pytest.raises(KeyError):
        ref["nonexistent"]


def test_entity_reference_addition():
    """Test combining EntityReferences."""
    ref1 = EntityReference({"dataset1": frozenset({"1", "2"})})
    ref2 = EntityReference(
        {"dataset1": frozenset({"2", "3"}), "dataset2": frozenset({"A"})}
    )
    combined = ref1 + ref2
    assert combined["dataset1"] == frozenset({"1", "2", "3"})
    assert combined["dataset2"] == frozenset({"A"})


def test_entity_reference_subset():
    """Test subset relationships between EntityReferences."""
    subset = EntityReference({"dataset1": frozenset({"1", "2"})})
    superset = EntityReference(
        {"dataset1": frozenset({"1", "2", "3"}), "dataset2": frozenset({"A"})}
    )

    assert subset <= superset
    assert not superset <= subset


def test_cluster_entity_creation():
    """Test basic ClusterEntity functionality."""
    ref = EntityReference({"dataset1": frozenset({"1", "2"})})
    entity = ClusterEntity(source_pks=ref)

    assert entity.source_pks == ref
    assert isinstance(entity.id, int)


def test_cluster_entity_addition():
    """Test combining ClusterEntity objects."""
    entity1 = ClusterEntity(source_pks=EntityReference({"dataset1": frozenset({"1"})}))
    entity2 = ClusterEntity(source_pks=EntityReference({"dataset1": frozenset({"2"})}))

    combined = entity1 + entity2
    assert combined.source_pks["dataset1"] == frozenset({"1", "2"})


def test_source_entity_creation():
    """Test basic SourceEntity functionality."""
    base_values = {"name": "John", "age": 30}
    ref = EntityReference({"dataset1": frozenset({"1", "2"})})

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


@pytest.mark.parametrize(
    (
        "probabilities",
        "left_clusters",
        "right_clusters",
        "threshold",
        "expected_count",
    ),
    [
        pytest.param(
            pa.table(
                {
                    "left_id": [1, 2],
                    "right_id": [2, 3],
                    "probability": [90, 85],
                }
            ),
            (
                make_cluster_entity(1, "test", ["a1"]),
                make_cluster_entity(2, "test", ["a2"]),
                make_cluster_entity(3, "test", ["a3"]),
            ),
            None,
            80,
            1,  # One merged entity containing all three records
            id="basic_dedupe_chain",
        ),
        pytest.param(
            pa.table(
                {
                    "left_id": [1],
                    "right_id": [4],
                    "probability": [95],
                }
            ),
            (make_cluster_entity(1, "left", ["a1"]),),
            (make_cluster_entity(4, "right", ["b1"]),),
            0.9,
            1,  # One merged entity from the link
            id="basic_link_match",
        ),
        pytest.param(
            pa.table(
                {
                    "left_id": [1, 2],
                    "right_id": [2, 3],
                    "probability": [75, 70],
                }
            ),
            (
                make_cluster_entity(1, "test", ["a1"]),
                make_cluster_entity(2, "test", ["a2"]),
                make_cluster_entity(3, "test", ["a3"]),
            ),
            None,
            80,
            3,  # No merging due to threshold
            id="threshold_prevents_merge",
        ),
        pytest.param(
            pa.table(
                {
                    "left_id": [],
                    "right_id": [],
                    "probability": [],
                }
            ),
            (
                make_cluster_entity(1, "test", ["a1"]),
                make_cluster_entity(2, "test", ["a2"]),
            ),
            None,
            80,
            2,  # No merging with empty probabilities
            id="empty_probabilities",
        ),
    ],
)
def test_probabilities_to_results_entities(
    probabilities: pa.Table,
    left_clusters: tuple[ClusterEntity, ...],
    right_clusters: tuple[ClusterEntity, ...] | None,
    threshold: float,
    expected_count: int,
) -> None:
    """Test probabilities_to_results_entities with various scenarios."""
    result = probabilities_to_results_entities(
        probabilities=probabilities,
        left_clusters=left_clusters,
        right_clusters=right_clusters,
        threshold=threshold,
    )

    assert len(result) == expected_count

    # For merging cases, verify all input entities are contained in the output
    all_inputs = set(left_clusters)
    if right_clusters:
        all_inputs.update(right_clusters)

    for input_entity in all_inputs:
        # Each input entity should be contained within one of the output entities
        assert any(input_entity in output_entity for output_entity in result)


@pytest.mark.parametrize(
    ("expected", "actual", "verbose", "want_identical", "want_result"),
    [
        # Identical sets
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2"])],
            [make_cluster_entity(1, "d1", ["1", "2"])],
            False,
            True,
            {},
            id="identical_sets",
        ),
        # Completely missing entity
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2"])],
            [],
            True,
            False,
            {
                "mean_similarity": 0.0,
                "partial": [],
                "missing": [{"id": 1, "source_pks": {"d1": frozenset(["1", "2"])}}],
                "extra": [],
            },
            id="completely_missing_entity",
        ),
        # Extra entity
        pytest.param(
            [],
            [make_cluster_entity(2, "d1", ["1", "2"])],
            True,
            False,
            {
                "mean_similarity": 0.0,
                "partial": [],
                "missing": [],
                "extra": [{"id": 2, "source_pks": {"d1": frozenset(["1", "2"])}}],
            },
            id="extra_entity",
        ),
        # Partial match
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2", "3"])],
            [make_cluster_entity(2, "d1", ["1", "2", "4"])],
            True,
            False,
            {
                # Jaccard similarity: |intersection| / |union| = 2 / 4 = 0.5
                "mean_similarity": 0.5,
                "partial": [
                    {
                        "missing_entity_id": 1,
                        "matches": [
                            {
                                "actual_entity_id": 2,
                                "similarity": 0.5,  # 2 common keys out of 4 total
                                "missing_pks": {"d1": frozenset(["3"])},
                                "extra_pks": {"d1": frozenset(["4"])},
                            }
                        ],
                    }
                ],
                "missing": [],
                "extra": [],
            },
            id="partial_match",
        ),
        # Complex scenario - partial match, missing, and extra
        pytest.param(
            [
                make_cluster_entity(1, "d1", ["1", "2"]),
                make_cluster_entity(2, "d1", ["3", "4"]),
                make_cluster_entity(3, "d1", ["5", "6"]),
            ],
            [
                make_cluster_entity(4, "d1", ["1", "7"]),
                make_cluster_entity(5, "d1", ["8", "9"]),
            ],
            True,
            False,
            {
                # Mean of best matches for all missing and extra entities:
                # Best matches: [1/3, 0, 0, 1/3, 0] = 2/15 = 0.1333...
                "mean_similarity": pytest.approx(2 / 15, rel=1e-2),
                "partial": [
                    {
                        "missing_entity_id": 1,
                        "matches": [
                            {
                                "actual_entity_id": 4,
                                "similarity": 1 / 3,  # 1 common key out of 3 total
                                "missing_pks": {"d1": frozenset(["2"])},
                                "extra_pks": {"d1": frozenset(["7"])},
                            }
                        ],
                    }
                ],
                "missing": [
                    {"id": 2, "source_pks": {"d1": frozenset(["3", "4"])}},
                    {"id": 3, "source_pks": {"d1": frozenset(["5", "6"])}},
                ],
                "extra": [{"id": 5, "source_pks": {"d1": frozenset(["8", "9"])}}],
            },
            id="complex_scenario",
        ),
        # Non-verbose mode (only shows mean_similarity)
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2", "3"])],
            [make_cluster_entity(2, "d1", ["1", "2", "4"])],
            False,
            False,
            {
                # Jaccard similarity: |intersection| / |union| = 2 / 4 = 0.5
                "mean_similarity": 0.5,
                "partial": [],
                "missing": [],
                "extra": [],
            },
            id="non_verbose_mode",
        ),
        # Mixed identical and different entities
        pytest.param(
            [
                make_cluster_entity(
                    1, "d1", ["1", "2"]
                ),  # Identical entity in both sets
                make_cluster_entity(2, "d1", ["3", "4"]),  # Will be partially matched
            ],
            [
                make_cluster_entity(1, "d1", ["1", "2"]),  # Same as in expected
                make_cluster_entity(3, "d1", ["3", "5"]),  # Partial match with entity 2
            ],
            True,
            False,
            {
                # Mean similarity combines perfect (1.0) and two partial (1/3) matches
                # Calculated as (1.0 + 1/3 + 1/3) / 3 = (5/9)
                "mean_similarity": pytest.approx(5 / 9, rel=1e-2),
                "partial": [
                    {
                        "missing_entity_id": 2,
                        "matches": [
                            {
                                "actual_entity_id": 3,
                                "similarity": 1 / 3,  # 1 common key out of 3 total
                                "missing_pks": {"d1": frozenset(["4"])},
                                "extra_pks": {"d1": frozenset(["5"])},
                            }
                        ],
                    }
                ],
                "missing": [],
                "extra": [],
            },
            id="mixed_identical_and_different",
        ),
    ],
)
def test_diff_results(
    expected: list[ClusterEntity],
    actual: list[ClusterEntity],
    verbose: bool,
    want_identical: bool,
    want_result: dict[str, Any],
):
    """Test diff_results function handles various scenarios correctly."""
    got_identical, got_result = diff_results(expected, actual, verbose)

    assert got_identical == want_identical

    if got_identical:
        assert got_result == {}
    else:
        # Handle complex nested dictionary comparison
        assert got_result["mean_similarity"] == pytest.approx(
            want_result["mean_similarity"], rel=1e-2
        )

        # Test partial matches
        assert len(got_result["partial"]) == len(want_result["partial"])
        for got_partial, want_partial in zip(
            got_result["partial"], want_result["partial"], strict=True
        ):
            assert got_partial["missing_entity_id"] == want_partial["missing_entity_id"]

            for got_match, want_match in zip(
                got_partial["matches"], want_partial["matches"], strict=True
            ):
                assert got_match["actual_entity_id"] == want_match["actual_entity_id"]
                assert got_match["similarity"] == pytest.approx(
                    want_match["similarity"], rel=1e-2
                )
                assert got_match["missing_pks"] == want_match["missing_pks"]
                assert got_match["extra_pks"] == want_match["extra_pks"]

        # Test missing and extra entities
        assert len(got_result["missing"]) == len(want_result["missing"])
        for got_missing, want_missing in zip(
            sorted(got_result["missing"], key=lambda x: x["id"]),
            sorted(want_result["missing"], key=lambda x: x["id"]),
            strict=True,
        ):
            assert got_missing["id"] == want_missing["id"]
            assert got_missing["source_pks"] == want_missing["source_pks"]

        assert len(got_result["extra"]) == len(want_result["extra"])
        for got_extra, want_extra in zip(
            sorted(got_result["extra"], key=lambda x: x["id"]),
            sorted(want_result["extra"], key=lambda x: x["id"]),
            strict=True,
        ):
            assert got_extra["id"] == want_extra["id"]
            assert got_extra["source_pks"] == want_extra["source_pks"]


def test_source_to_results_conversion():
    """Test converting source entities to cluster entities and comparing them."""
    # Create source entity present in multiple datasets
    source = SourceEntity(
        base_values={"name": "Test"},
        source_pks=EntityReference(
            {"dataset1": frozenset({"1", "2"}), "dataset2": frozenset({"A", "B"})}
        ),
    )

    # Convert different subsets to cluster entities
    results1 = source.to_cluster_entity("dataset1")
    results2 = source.to_cluster_entity("dataset1", "dataset2")
    results3 = source.to_cluster_entity("dataset2")

    # Test different comparison scenarios
    identical, report = diff_results([results1], [results1])
    assert identical
    assert report == {}

    # Compare partial overlap
    identical, report = diff_results([results1], [results2])
    assert not identical
    assert "dataset2" in str(results2 - results1)

    # Compare disjoint sets
    identical, report = diff_results([results1], [results3])
    assert not identical
    assert results1.similarity_ratio(results3) == 0.0

    # Test missing dataset returns None
    assert source.to_cluster_entity("nonexistent") is None


@pytest.mark.parametrize(
    ("base_generator", "expected_type"),
    [
        pytest.param("name", "TEXT", id="text_generator"),
        pytest.param("random_int", "INTEGER", id="integer_generator"),
        pytest.param("date_this_decade", "DATE", id="date_generator"),
    ],
)
def test_feature_config_sql_type_inference(
    base_generator: str, expected_type: str
) -> None:
    """Test that SQL types are correctly inferred from feature configurations."""
    feature_config = FeatureConfig(name=base_generator, base_generator=base_generator)
    assert feature_config.sql_type == expected_type
