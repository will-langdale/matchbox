from math import comb
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pytest

from matchbox.common.factories.results import (
    calculate_min_max_edges,
    generate_dummy_probabilities,
    verify_components,
)
from matchbox.common.factories.sources import (
    FeatureConfig,
    ReplaceRule,
    SuffixRule,
    source_factory,
)


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
    p_left = probabilities["left"].to_pylist()
    p_right = probabilities["right"].to_pylist()

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
