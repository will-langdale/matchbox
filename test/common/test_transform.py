from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import chain
from typing import Any, Iterator
from unittest.mock import patch

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from matchbox.common.factories import generate_dummy_probabilities, verify_components
from matchbox.common.hash import IntMap
from matchbox.common.transform import (
    attach_components_to_probabilities,
    component_to_hierarchy,
    to_hierarchical_clusters,
)


def _combine_strings(self, *n: str) -> str:
    """
    Combine n strings into a single string, with a cache.
    Meant to replace `matchbox.common.hash.IntMap.index`

    Args:
        *args: Variable number of strings to combine

    Returns:
        A single string
    """
    value_set = frozenset(n)
    if value_set in self.mapping:
        return self.mapping[value_set]

    letters = set(chain.from_iterable(n))

    new_id = "".join(sorted(letters))
    self.mapping[value_set] = new_id
    return new_id


@contextmanager
def parallel_pool_for_tests(
    max_workers: int = 2, timeout: int = 30
) -> Iterator[ThreadPoolExecutor]:
    """Context manager for safe parallel execution in tests using threads.

    Args:
        max_workers: Maximum number of worker threads
        timeout: Maximum seconds to wait for each task
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            yield executor
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.parametrize(
    ("parameters"),
    [
        {
            "left_range": (0, 1_000),
            "right_range": (1_000, 2_000),
            "prob_range": (0.6, 0.8),
            "num_components": 10,
            "total_rows": 100_000,
        },
    ],
    ids=["simple"],
)
def test_probabilities_factory(parameters: dict[str, Any]):
    left_values = range(*parameters["left_range"])
    right_values = range(*parameters["right_range"])

    probabilities = generate_dummy_probabilities(
        left_values=left_values,
        right_values=right_values,
        prob_range=parameters["prob_range"],
        num_components=parameters["num_components"],
        total_rows=parameters["total_rows"],
    )
    report = verify_components(table=probabilities)

    assert report["num_components"] == parameters["num_components"]
    assert set(pc.unique(probabilities["left"]).to_pylist()) == set(left_values)
    assert set(pc.unique(probabilities["right"]).to_pylist()) == set(right_values)
    assert (
        pc.max(probabilities["probability"]).as_py() / 100
        <= parameters["prob_range"][1]
    )
    assert (
        pc.min(probabilities["probability"]).as_py() / 100
        >= parameters["prob_range"][0]
    )


@pytest.mark.parametrize(
    ("parameters"),
    [
        {
            "left_range": (0, 1_000),
            "right_range": (1_000, 2_000),
            "prob_range": (0.6, 0.8),
            "num_components": 10,
            "total_rows": 100_000,
        },
    ],
    ids=["simple"],
)
def test_attach_components_to_probabilities(parameters: dict[str, Any]):
    left_values = range(*parameters["left_range"])
    right_values = range(*parameters["right_range"])

    probabilities = generate_dummy_probabilities(
        left_values=left_values,
        right_values=right_values,
        prob_range=parameters["prob_range"],
        num_components=parameters["num_components"],
        total_rows=parameters["total_rows"],
    )

    with_components = attach_components_to_probabilities(probabilities=probabilities)

    assert len(pc.unique(with_components["component"])) == parameters["num_components"]


@pytest.mark.parametrize(
    ("probabilities", "hierarchy"),
    [
        # Test case 1: Equal probabilities
        (
            {
                "left": ["a", "b", "c"],
                "right": ["b", "c", "d"],
                "probability": [100, 100, 100],
            },
            {
                ("ab", "a", 100),
                ("ab", "b", 100),
                ("bc", "b", 100),
                ("bc", "c", 100),
                ("cd", "c", 100),
                ("cd", "d", 100),
                ("abcd", "ab", 100),
                ("abcd", "bc", 100),
                ("abcd", "cd", 100),
            },
        ),
        # Test case 2: Asymmetric probabilities
        (
            {
                "left": ["w", "x", "y"],
                "right": ["x", "y", "z"],
                "probability": [90, 85, 80],
            },
            {
                ("wx", "w", 90),
                ("wx", "x", 90),
                ("xy", "x", 85),
                ("xy", "y", 85),
                ("wxy", "wx", 85),
                ("wxy", "xy", 85),
                ("yz", "y", 80),
                ("yz", "z", 80),
                ("wxyz", "wxy", 80),
                ("wxyz", "yz", 80),
            },
        ),
        # Test case 3: Single two-item component
        (
            {
                "left": ["x"],
                "right": ["y"],
                "probability": [90],
            },
            {
                ("xy", "x", 90),
                ("xy", "y", 90),
            },
        ),
        # Test case 4: A component larger than two remains unchanged
        # at a successive threshold
        (
            {
                "left": ["x", "y", "a"],
                "right": ["y", "z", "b"],
                "probability": [90, 90, 85],
            },
            {
                ("xy", "x", 90),
                ("xy", "y", 90),
                ("yz", "y", 90),
                ("yz", "z", 90),
                ("xyz", "xy", 90),
                ("xyz", "yz", 90),
                ("ab", "a", 85),
                ("ab", "b", 85),
            },
        ),
    ],
    ids=["equal", "asymmetric", "single", "unchanged"],
)
def test_component_to_hierarchy(
    probabilities: dict[str, list[str | float]], hierarchy: set[tuple[str, str, int]]
):
    with patch.object(IntMap, "index", _combine_strings):
        probabilities_table = (
            pa.Table.from_pydict(probabilities)
            .cast(
                pa.schema(
                    [
                        ("left", pa.string()),
                        ("right", pa.string()),
                        ("probability", pa.uint8()),
                    ]
                )
            )
            .sort_by([("probability", "descending")])
        )

        parents, children, probs = zip(*hierarchy, strict=False)

        hierarchy_true = (
            pa.table(
                [parents, children, probs], names=["parent", "child", "probability"]
            )
            .cast(
                pa.schema(
                    [
                        ("parent", pa.string()),
                        ("child", pa.string()),
                        ("probability", pa.uint8()),
                    ]
                )
            )
            .sort_by(
                [
                    ("probability", "descending"),
                    ("parent", "ascending"),
                    ("child", "ascending"),
                ]
            )
            .filter(pc.is_valid(pc.field("parent")))
        )

        hierarchy = component_to_hierarchy(
            probabilities_table, salt=1, dtype=pa.string
        ).sort_by(
            [
                ("probability", "descending"),
                ("parent", "ascending"),
                ("child", "ascending"),
            ]
        )

        assert hierarchy.equals(hierarchy_true)


@pytest.mark.parametrize(
    ("input_data", "expected_hierarchy"),
    [
        # Single component test case
        (
            {
                "component": [1, 1, 1],
                "left": ["a", "b", "c"],
                "right": ["b", "c", "d"],
                "probability": [90, 85, 80],
            },
            {
                ("ab", "a", 90),
                ("ab", "b", 90),
                ("bc", "b", 85),
                ("bc", "c", 85),
                ("abc", "ab", 85),
                ("abc", "bc", 85),
                ("cd", "c", 80),
                ("cd", "d", 80),
                ("abcd", "abc", 80),
                ("abcd", "cd", 80),
            },
        ),
        # Multiple components test case
        (
            {
                "component": [1, 1, 2, 2],
                "left": ["a", "b", "x", "y"],
                "right": ["b", "c", "y", "z"],
                "probability": [90, 85, 95, 92],
            },
            {
                ("xy", "x", 95),
                ("xy", "y", 95),
                ("yz", "y", 92),
                ("yz", "z", 92),
                ("xyz", "xy", 92),
                ("xyz", "yz", 92),
                ("ab", "a", 90),
                ("ab", "b", 90),
                ("bc", "b", 85),
                ("bc", "c", 85),
                ("abc", "ab", 85),
                ("abc", "bc", 85),
            },
        ),
    ],
    ids=["single_component", "multiple_components"],
)
def test_hierarchical_clusters(input_data, expected_hierarchy):
    # Create input table
    probabilities = pa.table(
        input_data,
        schema=pa.schema(
            [
                ("component", pa.uint64()),
                ("left", pa.string()),
                ("right", pa.string()),
                ("probability", pa.uint8()),
            ]
        ),
    )

    # Create expected output table
    parents, children, probs = zip(*expected_hierarchy, strict=False)
    expected = (
        pa.table([parents, children, probs], names=["parent", "child", "probability"])
        .cast(
            pa.schema(
                [
                    ("parent", pa.string()),
                    ("child", pa.string()),
                    ("probability", pa.uint8()),
                ]
            )
        )
        .sort_by(
            [
                ("probability", "descending"),
                ("parent", "ascending"),
                ("child", "ascending"),
            ]
        )
    )

    # Run and compare
    with (
        patch(
            "matchbox.common.transform.ProcessPoolExecutor",
            lambda *args, **kwargs: parallel_pool_for_tests(timeout=30),
        ),
        patch.object(IntMap, "index", side_effect=_combine_strings),
    ):
        result = to_hierarchical_clusters(
            probabilities, dtype=pa.string, proc_func=component_to_hierarchy
        )

    result = result.sort_by(
        [
            ("probability", "descending"),
            ("parent", "ascending"),
            ("child", "ascending"),
        ]
    )

    assert result.schema == expected.schema
    assert result.equals(expected)
