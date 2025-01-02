from typing import Any

import pyarrow.compute as pc
import pytest

from matchbox.common.factories import generate_dummy_probabilities, verify_components


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


def test_dummy_probs_min_edges_dedupe(): ...
