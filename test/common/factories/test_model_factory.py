from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from matchbox.common.arrow import SCHEMA_RESULTS
from matchbox.common.dtos import ModelType
from matchbox.common.factories.entities import (
    EntityReference,
    ResultsEntity,
    SourceEntity,
)
from matchbox.common.factories.models import (
    calculate_min_max_edges,
    generate_dummy_probabilities,
    generate_entity_probabilities,
    model_factory,
    validate_components,
    verify_components,
)
from matchbox.common.factories.sources import SourceDummy, linked_sources_factory


def test_model_factory_default():
    """Test that model_factory generates a dummy model with default parameters."""
    dummy = model_factory()

    assert len(dummy.to_components()) == 10
    assert dummy.model.metadata.type == ModelType.DEDUPER
    assert dummy.model.metadata.right_resolution is None

    # Check that probabilities table was generated correctly
    assert len(dummy.data) > 0
    assert dummy.data.schema.equals(SCHEMA_RESULTS)


def test_model_factory_with_custom_params():
    """Test model_factory with custom parameters."""
    name = "test_model"
    description = "test description"
    n_true_entities = 5
    prob_range = (0.9, 1.0)

    dummy = model_factory(
        name=name,
        description=description,
        n_true_entities=n_true_entities,
        prob_range=prob_range,
    )

    assert dummy.model.metadata.name == name
    assert dummy.model.metadata.description == description
    assert len(dummy.to_components()) == n_true_entities


@pytest.mark.parametrize(
    ("model_type"),
    [
        pytest.param("deduper", id="deduper"),
        pytest.param("linker", id="linker"),
    ],
)
def test_model_factory_different_types(model_type: str):
    """Test model_factory handles different model types correctly."""
    dummy = model_factory(model_type=model_type)

    assert dummy.model.metadata.type == model_type

    if model_type == ModelType.LINKER:
        assert dummy.model.metadata.right_resolution is not None

        # Check no collisions
        left_ids = set(dummy.data["left_id"].to_pylist())
        right_ids = set(dummy.data["right_id"].to_pylist())
        assert len(left_ids.intersection(right_ids)) == 0


@pytest.mark.parametrize(
    ("seed1", "seed2", "should_be_equal"),
    [
        pytest.param(42, 42, True, id="same_seeds"),
        pytest.param(1, 2, False, id="different_seeds"),
    ],
)
def test_model_factory_seed_behavior(seed1: int, seed2: int, should_be_equal: bool):
    """Test that model_factory handles seeds correctly for reproducibility."""
    dummy1 = model_factory(seed=seed1)
    dummy2 = model_factory(seed=seed2)

    if should_be_equal:
        assert dummy1.model.metadata.name == dummy2.model.metadata.name
        assert dummy1.model.metadata.description == dummy2.model.metadata.description
        assert dummy1.data.equals(dummy2.data)
    else:
        assert dummy1.model.metadata.name != dummy2.model.metadata.name
        assert dummy1.model.metadata.description != dummy2.model.metadata.description
        assert not dummy1.data.equals(dummy2.data)


@pytest.mark.parametrize(
    ("left_source_type", "right_source_type"),
    [
        pytest.param(pa.Table, pa.Table, id="both_pyarrow"),
        pytest.param(SourceDummy, SourceDummy, id="both_sourcedummy"),
        pytest.param(pa.Table, SourceDummy, id="mixed_sources"),
    ],
)
def test_model_factory_with_provided_sources(left_source_type, right_source_type):
    """Test model_factory handles different source input types correctly."""
    # Setup mock sources
    linked = linked_sources_factory()
    left_dummy = linked.sources.get("crn")
    right_dummy = linked.sources.get("cdms")

    if left_source_type == pa.Table:
        left_source = left_dummy.data
    else:
        left_source = left_dummy

    if right_source_type == pa.Table:
        right_source = right_dummy.data
    else:
        right_source = right_dummy

    # Create model
    dummy = model_factory(left_source=left_source, right_source=right_source)

    # Assert it worked
    assert dummy.model.metadata.type == ModelType.LINKER
    assert dummy.model.metadata.left_resolution is not None
    assert dummy.model.metadata.right_resolution is not None
    assert len(dummy.data) > 0


@pytest.mark.parametrize(
    ("left_n", "right_n", "n_components", "true_min", "true_max"),
    [
        (10, None, 2, 8, 20),
        (11, None, 2, 9, 25),
        (9, 9, 3, 15, 27),
        (8, 4, 3, 9, 11),
        (4, 8, 3, 9, 11),
        (8, 8, 3, 13, 22),
    ],
    ids=[
        "dedupe_no_mod",
        "dedup_mod",
        "link_no_mod",
        "link_left_mod",
        "link_right_mod",
        "link_same_mod",
    ],
)
def test_calculate_min_max_edges(
    left_n: int,
    right_n: int | None,
    n_components: int,
    true_min: int,
    true_max: int,
):
    deduplicate = False
    if not right_n:
        deduplicate = True
        right_n = left_n
    min_edges, max_edges = calculate_min_max_edges(
        left_n, right_n, n_components, deduplicate
    )

    assert true_min == min_edges
    assert true_max == max_edges


@pytest.mark.parametrize(
    ("parameters"),
    [
        {
            "left_count": 5,
            "right_count": None,
            "prob_range": (0.6, 0.8),
            "num_components": 3,
            "total_rows": 2,
        },
        {
            "left_count": 1000,
            "right_count": None,
            "prob_range": (0.6, 0.8),
            "num_components": 10,
            "total_rows": calculate_min_max_edges(1000, 1000, 10, True)[0],
        },
        {
            "left_count": 1_000,
            "right_count": None,
            "prob_range": (0.6, 0.8),
            "num_components": 10,
            "total_rows": calculate_min_max_edges(1000, 1000, 10, True)[1],
        },
        {
            "left_count": 1_000,
            "right_count": 1_000,
            "prob_range": (0.6, 0.8),
            "num_components": 10,
            "total_rows": calculate_min_max_edges(1000, 1000, 10, False)[0],
        },
        {
            "left_count": 1_000,
            "right_count": 1_000,
            "prob_range": (0.6, 0.8),
            "num_components": 10,
            "total_rows": calculate_min_max_edges(1000, 1000, 10, False)[1],
        },
    ],
    ids=[
        "dedupe_no_edges",
        "dedupe_min",
        "dedupe_max",
        "link_min",
        "link_max",
    ],
)
def test_generate_dummy_probabilities(parameters: dict[str, Any]):
    len_left = parameters["left_count"]
    len_right = parameters["right_count"]
    if len_right:
        total_len = len_left + len_right
        len_right = parameters["right_count"]
        rand_vals = np.random.choice(a=total_len, replace=False, size=total_len)
        left_values = tuple(rand_vals[:len_left])
        right_values = tuple(rand_vals[len_left:])
    else:
        rand_vals = np.random.choice(a=len_left, replace=False, size=len_left)
        left_values = tuple(rand_vals[:len_left])
        right_values = None

    n_components = parameters["num_components"]
    total_rows = parameters["total_rows"]

    probabilities = generate_dummy_probabilities(
        left_values=left_values,
        right_values=right_values,
        prob_range=parameters["prob_range"],
        num_components=n_components,
        total_rows=total_rows,
    )
    report = verify_components(table=probabilities, all_nodes=rand_vals)
    p_left = probabilities["left_id"].to_pylist()
    p_right = probabilities["right_id"].to_pylist()

    assert report["num_components"] == n_components

    # Link job
    if right_values:
        assert set(p_left) <= set(left_values)
        assert set(p_right) <= set(right_values)
    # Dedupe
    else:
        assert set(p_left) | set(p_right) <= set(left_values)

    assert (
        pc.max(probabilities["probability"]).as_py() / 100
        <= parameters["prob_range"][1]
    )
    assert (
        pc.min(probabilities["probability"]).as_py() / 100
        >= parameters["prob_range"][0]
    )

    assert len(probabilities) == total_rows

    edges = zip(p_left, p_right, strict=True)
    edges_set = {tuple(sorted(e)) for e in edges}
    assert len(edges_set) == total_rows

    self_references = [e for e in edges if e[0] == e[1]]
    assert len(self_references) == 0


@pytest.mark.parametrize(
    ("parameters"),
    [
        {
            "left_range": (0, 10_000),
            "right_range": (10_000, 20_000),
            "num_components": 2,
            "total_rows": 1,
        },
        {
            "left_range": (0, 10),
            "right_range": (10, 20),
            "num_components": 2,
            "total_rows": 8_000,
        },
    ],
    ids=["lower_than_min", "higher_than_max"],
)
def test_generate_dummy_probabilities_errors(parameters: dict[str, Any]):
    left_values = tuple(range(*parameters["left_range"]))
    right_values = tuple(range(*parameters["right_range"]))

    with pytest.raises(ValueError):
        generate_dummy_probabilities(
            left_values=left_values,
            right_values=right_values,
            prob_range=(0.6, 0.8),
            num_components=parameters["num_components"],
            total_rows=parameters["total_rows"],
        )


def _make_results_entity(id: int, dataset: str, pks: list[str]) -> ResultsEntity:
    """Helper to create a ResultsEntity with specified dataset and PKs."""
    return ResultsEntity(
        id=id,
        source_pks=EntityReference(mapping=frozenset([(dataset, frozenset(pks))])),
    )


def _make_source_entity(dataset: str, pks: list[str], base_val: str) -> SourceEntity:
    """Helper to create a SourceEntity with specified dataset and PKs."""
    entity = SourceEntity(base_values={"name": base_val})
    entity.add_source_reference(dataset, pks)
    return entity


@pytest.mark.parametrize(
    (
        "left_entities",
        "right_entities",
        "source_entities",
        "prob_range",
        "expected_edge_count",
    ),
    [
        # Basic deduplication case - two results entities matching one source entity
        pytest.param(
            frozenset(
                {
                    _make_results_entity(1, "test", ["a1"]),
                    _make_results_entity(2, "test", ["a2"]),
                }
            ),
            None,
            frozenset({_make_source_entity("test", ["a1", "a2"], "a")}),
            (0.8, 1.0),
            1,  # Should generate one edge with ID 1 < 2
            id="basic_dedupe",
        ),
        # Basic linking case - distinct left and right entities
        pytest.param(
            frozenset({_make_results_entity(1, "left", ["a1"])}),
            frozenset({_make_results_entity(2, "right", ["b1"])}),
            frozenset(
                {
                    _make_source_entity("left", ["a1"], "a"),
                    _make_source_entity("right", ["b1"], "b"),
                }
            ),
            (0.8, 1.0),
            0,  # No edges since entities belong to different source entities
            id="basic_link",
        ),
        # Successful linking case
        pytest.param(
            frozenset({_make_results_entity(1, "test", ["a1"])}),
            frozenset({_make_results_entity(2, "test", ["a2"])}),
            frozenset({_make_source_entity("test", ["a1", "a2"], "a")}),
            (0.8, 1.0),
            1,
            id="successful_link",
        ),
        # Linking case with multiple components
        pytest.param(
            frozenset(
                {
                    _make_results_entity(1, "test", ["a1"]),
                    _make_results_entity(2, "test", ["b1"]),
                }
            ),
            frozenset(
                {
                    _make_results_entity(3, "test", ["a2"]),
                    _make_results_entity(4, "test", ["b2"]),
                }
            ),
            frozenset(
                {
                    _make_source_entity("test", ["a1", "a2"], "a"),
                    _make_source_entity("test", ["b1", "b2"], "b"),
                }
            ),
            (0.8, 1.0),
            2,  # Should connect 1-3 and 2-4
            id="multi_component_link",
        ),
        # Empty input sets
        pytest.param(
            frozenset(), frozenset(), frozenset(), (0.8, 1.0), 0, id="empty_sets"
        ),
    ],
)
def test_generate_entity_probabilities(
    left_entities: frozenset[ResultsEntity],
    right_entities: frozenset[ResultsEntity] | None,
    source_entities: frozenset[SourceEntity],
    prob_range: tuple[float, float],
    expected_edge_count: list[tuple[int, int]],
):
    """Test generate_entity_probabilities with various scenarios."""
    # Run the function
    result = generate_entity_probabilities(
        left_entities, right_entities, source_entities, prob_range
    )

    # Get edges from result
    edges = list(
        zip(
            result.column("left_id").to_pylist(),
            result.column("right_id").to_pylist(),
            strict=True,
        )
    )

    # Check number of edges matches expected
    assert len(edges) == expected_edge_count

    # For non-empty results, validate components
    if edges:
        # Get all entities (combine left and right for linking case)
        all_entities = left_entities | (
            right_entities if right_entities is not None else set()
        )

        # Validate that components are correct
        assert validate_components(edges, all_entities, source_entities)

        # Check probability ranges
        probs = result.column("probability").to_numpy()
        prob_min, prob_max = int(prob_range[0] * 100), int(prob_range[1] * 100)
        assert all(prob_min <= p <= prob_max for p in probs)

    # Check schema
    assert result.schema.equals(SCHEMA_RESULTS)
