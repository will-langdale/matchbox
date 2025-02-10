from math import comb
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pytest
from sqlalchemy import create_engine

from matchbox.common.arrow import SCHEMA_INDEX, SCHEMA_RESULTS
from matchbox.common.dtos import ModelType
from matchbox.common.factories.models import (
    calculate_min_max_edges,
    generate_dummy_probabilities,
    model_factory,
    verify_components,
)
from matchbox.common.factories.sources import (
    FeatureConfig,
    ReplaceRule,
    SuffixRule,
    source_factory,
)
from matchbox.common.sources import SourceAddress


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
    left_n: int, right_n: int | None, n_components: int, true_min: int, true_max: int
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
        left_values = list(rand_vals[:len_left])
        right_values = list(rand_vals[len_left:])
    else:
        rand_vals = np.random.choice(a=len_left, replace=False, size=len_left)
        left_values = list(rand_vals[:len_left])
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
    left_values = range(*parameters["left_range"])
    right_values = range(*parameters["right_range"])

    with pytest.raises(ValueError):
        generate_dummy_probabilities(
            left_values=left_values,
            right_values=right_values,
            prob_range=(0.6, 0.8),
            num_components=parameters["num_components"],
            total_rows=parameters["total_rows"],
        )


def test_source_factory_default():
    """Test that source_factory generates a dummy source with default parameters."""
    source = source_factory()

    assert source.metrics.n_true_entities == 10
    assert source.data_hashes.schema.equals(SCHEMA_INDEX)


def test_source_factory_metrics_match_data_basic():
    """Test that the metrics match the actual data content for a simple case."""
    features = [
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=[
                SuffixRule(suffix=" Inc"),
                SuffixRule(suffix=" Ltd"),
            ],
        ),
    ]

    n_true_entities = 5
    dummy_source = source_factory(
        n_true_entities=n_true_entities, repetition=1, features=features, seed=42
    )

    # Check true entities count
    unique_base_values = len(
        set(
            dummy_source.data.to_pandas()
            .assign(
                company_name=lambda x: x["company_name"]
                .str.replace(" Inc", "")
                .str.replace(" Ltd", "")
            )
            .groupby("pk")["company_name"]
            .first()
        )
    )
    assert unique_base_values == n_true_entities

    # Check variations per entity
    variations_per_entity = len(features[0].variations) + 1  # +1 for base value
    assert dummy_source.metrics.n_unique_rows == variations_per_entity

    # Verify total rows
    expected_total_rows = n_true_entities * variations_per_entity
    actual_total_rows = len(dummy_source.data)
    assert actual_total_rows == expected_total_rows

    # Verify potential pairs calculation
    expected_pairs = comb(variations_per_entity, 2) * n_true_entities
    assert dummy_source.metrics.n_potential_pairs == expected_pairs


def test_source_factory_metrics_with_repetition():
    """Test that repetition properly multiplies the data with correct metrics."""
    features = [
        FeatureConfig(
            name="email",
            base_generator="email",
            variations=[ReplaceRule(old="@", new="+test@")],
        ),
    ]

    n_true_entities = 3
    repetition = 2
    dummy_source = source_factory(
        n_true_entities=n_true_entities,
        repetition=repetition,
        features=features,
        seed=42,
    )

    # Base metrics should not be affected by repetition
    assert dummy_source.metrics.n_true_entities == n_true_entities
    assert dummy_source.metrics.n_unique_rows == 2  # base + 1 variation

    # But total rows should be multiplied
    expected_total_rows = (
        n_true_entities * 2 * repetition
    )  # 2 rows per entity * repetition
    actual_total_rows = len(dummy_source.data)
    assert actual_total_rows == expected_total_rows


def test_source_factory_metrics_with_multiple_features():
    """Test that metrics are correct when multiple features have multiple variations."""
    features = [
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=[
                SuffixRule(suffix=" Inc"),
                SuffixRule(suffix=" Ltd"),
            ],
        ),
        FeatureConfig(
            name="email",
            base_generator="email",
            variations=[ReplaceRule(old="@", new="+test@")],
        ),
    ]

    n_true_entities = 4
    dummy_source = source_factory(
        n_true_entities=n_true_entities, repetition=1, features=features, seed=42
    )

    # Should use max variations across features
    max_variations = max(len(f.variations) for f in features)
    expected_rows_per_entity = max_variations + 1  # +1 for base value

    assert dummy_source.metrics.n_unique_rows == expected_rows_per_entity
    assert len(dummy_source.data) == n_true_entities * expected_rows_per_entity


def test_source_factory_data_hashes_integrity():
    """Test that data_hashes correctly identifies identical rows."""
    features = [
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=[SuffixRule(suffix=" Inc")],
        ),
    ]

    n_true_entities = 3
    repetition = 2
    dummy_source = source_factory(
        n_true_entities=n_true_entities,
        repetition=repetition,
        features=features,
        seed=42,
    )

    # Convert to pandas for easier analysis
    hashes_df = dummy_source.data_hashes.to_pandas()
    data_df = dummy_source.data.to_pandas()

    # For each hash group, verify that the corresponding rows are identical
    for _, group in hashes_df.groupby("hash"):
        pks = group["source_pk"].explode()
        rows = data_df[data_df["pk"].isin(pks)]

        # All rows in the same hash group should have identical feature values
        for feature in features:
            assert len(rows[feature.name].unique()) == 1

    # Due to repetition=2, each unique row should appear
    # in exactly one hash group with two PKs
    assert all(
        len(group["source_pk"].explode()) == repetition
        for _, group in hashes_df.groupby("hash")
    )

    # Total number of hash groups should equal
    # number of unique rows * number of true entities
    expected_hash_groups = n_true_entities * (
        len(features[0].variations) + 1
    )  # +1 for base value
    assert len(hashes_df["hash"].unique()) == expected_hash_groups


def test_source_dummy_to_mock():
    """Test that SourceDummy.to_mock() creates a correctly configured mock."""
    # Create a source dummy with some test data
    features = [
        FeatureConfig(
            name="test_field",
            base_generator="word",
            variations=[SuffixRule(suffix="_variant")],
        )
    ]

    dummy_source = source_factory(
        features=features, full_name="test.source", n_true_entities=2, seed=42
    )

    # Create the mock
    mock_source = dummy_source.to_mock()

    # Test that method calls are tracked
    mock_source.set_engine("test_engine")
    mock_source.default_columns()
    mock_source.hash_data()

    mock_source.set_engine.assert_called_once_with("test_engine")
    mock_source.default_columns.assert_called_once()
    mock_source.hash_data.assert_called_once()

    # Test method return valuse
    assert mock_source.set_engine("test_engine") == mock_source
    assert mock_source.default_columns() == mock_source
    assert mock_source.hash_data() == dummy_source.data_hashes

    # Test model dump methods
    original_dump = dummy_source.source.model_dump()
    mock_dump = mock_source.model_dump()
    assert mock_dump == original_dump

    original_json = dummy_source.source.model_dump_json()
    mock_json = mock_source.model_dump_json()
    assert mock_json == original_json

    # Verify side effect functions were set correctly
    mock_source.model_dump.assert_called_once()
    mock_source.model_dump_json.assert_called_once()

    # Test that to_table contains the correct data
    assert mock_source.to_table == dummy_source.data
    # Verify the number of rows matches what we expect from metrics
    assert (
        mock_source.to_table.shape[0]
        == dummy_source.metrics.n_true_entities * dummy_source.metrics.n_unique_rows
    )


def test_source_factory_mock_properties():
    """Test that source properties set in source_factory match generated Source."""
    # Create source with specific features and name
    features = [
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=[SuffixRule(suffix=" Ltd")],
        ),
        FeatureConfig(
            name="registration_id",
            base_generator="numerify",
            parameters={"text": "######"},
        ),
    ]

    full_name = "companies"
    engine = create_engine("sqlite:///:memory:")

    dummy_source = source_factory(
        features=features, full_name=full_name, engine=engine
    ).source

    # Check source address properties
    assert dummy_source.address.full_name == full_name

    # Warehouse hash should be consistent for same engine config
    expected_address = SourceAddress.compose(engine=engine, full_name=full_name)
    assert dummy_source.address.warehouse_hash == expected_address.warehouse_hash

    # Check column configuration
    assert len(dummy_source.columns) == len(features)
    for feature, column in zip(features, dummy_source.columns, strict=False):
        assert column.name == feature.name
        assert column.alias == feature.name
        assert column.type is None

    # Check default alias (should match full_name) and default pk
    assert dummy_source.alias == full_name
    assert dummy_source.db_pk == "pk"

    # Verify source properties are preserved through model_dump
    dump = dummy_source.model_dump()
    assert dump["address"]["full_name"] == full_name
    assert dump["columns"] == [
        {"name": f.name, "alias": f.name, "type": None} for f in features
    ]


def test_model_factory_default():
    """Test that model_factory generates a dummy model with default parameters."""
    model = model_factory()

    assert model.metrics.n_true_entities == 10
    assert model.model.type == ModelType.DEDUPER
    assert model.model.right_source is None

    # Check that probabilities table was generated correctly
    assert len(model.data) > 0
    assert model.data.schema.equals(SCHEMA_RESULTS)


def test_model_factory_with_custom_params():
    """Test model_factory with custom parameters."""
    name = "test_model"
    description = "test description"
    n_true_entities = 5
    prob_range = (0.9, 1.0)

    model = model_factory(
        name=name,
        description=description,
        n_true_entities=n_true_entities,
        prob_range=prob_range,
    )

    assert model.model.name == name
    assert model.model.description == description
    assert model.metrics.n_true_entities == n_true_entities

    # Check probability range
    probs = model.data.column("probability").to_pylist()
    assert all(90 <= p <= 100 for p in probs)


@pytest.mark.parametrize(
    ("model_type", "should_have_right_source"),
    [
        pytest.param("deduper", False, id="deduper"),
        pytest.param("linker", True, id="linker"),
    ],
)
def test_model_factory_different_types(model_type: str, should_have_right_source: bool):
    """Test model_factory handles different model types correctly."""
    model = model_factory(type=model_type)

    assert model.model.type == model_type
    assert (model.model.right_source is not None) == should_have_right_source

    if model_type == ModelType.LINKER:
        # Check that left and right values are in different ranges
        left_vals = model.data.column("left_id").to_pylist()
        right_vals = model.data.column("right_id").to_pylist()
        assert all(lv < rv for lv, rv in zip(left_vals, right_vals, strict=False))


@pytest.mark.parametrize(
    ("seed1", "seed2", "should_be_equal"),
    [
        pytest.param(42, 42, True, id="same_seeds"),
        pytest.param(1, 2, False, id="different_seeds"),
    ],
)
def test_model_factory_seed_behavior(seed1: int, seed2: int, should_be_equal: bool):
    """Test that model_factory handles seeds correctly for reproducibility."""
    model1 = model_factory(seed=seed1)
    model2 = model_factory(seed=seed2)

    if should_be_equal:
        assert model1.model.name == model2.model.name
        assert model1.model.description == model2.model.description
        assert model1.data.equals(model2.data)
    else:
        assert model1.model.name != model2.model.name
        assert not model1.data.equals(model2.data)
