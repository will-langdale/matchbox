from collections.abc import Mapping

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from matchbox.client.resolvers import (
    Components,
    ComponentsSettings,
    ResolverMethod,
    ResolverSettings,
    add_resolver_class,
    get_resolver_class,
)
from matchbox.common.dtos import ResolutionName


def test_components_settings_normalises_float_thresholds() -> None:
    settings = ComponentsSettings(thresholds={"model_a": 0.63})
    assert settings.thresholds == {"model_a": 63}


def test_components_compute_clusters_uses_thresholds() -> None:
    method = Components(
        settings=ComponentsSettings(thresholds={"model_a": 60}),
    )
    model_edges = {
        "model_a": pl.DataFrame(
            {
                "left_id": [1, 2],
                "right_id": [2, 3],
                "probability": [80, 40],
            },
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "probability": pl.UInt8,
            },
        )
    }
    resolver_assignments = {
        "resolver_a": pl.DataFrame(
            {
                "parent_id": [10, 10],
                "child_id": [3, 4],
            },
            schema={"parent_id": pl.UInt64, "child_id": pl.UInt64},
        )
    }

    clusters = method.compute_clusters(
        model_edges=model_edges,
        resolver_assignments=resolver_assignments,
    )

    expected = pl.DataFrame(
        {
            "parent_id": [1, 1, 2, 2],
            "child_id": [1, 2, 3, 4],
        },
        schema={"parent_id": pl.UInt64, "child_id": pl.UInt64},
    )
    assert_frame_equal(clusters, expected)


class _DummyResolverSettings(ResolverSettings):
    pass


class _DummyResolverMethod(ResolverMethod):
    settings: _DummyResolverSettings

    def compute_clusters(
        self,
        model_edges: Mapping[ResolutionName, pl.DataFrame],
        resolver_assignments: Mapping[ResolutionName, pl.DataFrame],
    ) -> pl.DataFrame:
        return pl.DataFrame(schema={"parent_id": pl.UInt64, "child_id": pl.UInt64})


def test_resolver_registry_allows_custom_class() -> None:
    add_resolver_class(_DummyResolverMethod)
    assert get_resolver_class("_DummyResolverMethod") is _DummyResolverMethod


def test_components_validate_settings_payload() -> None:
    with pytest.raises(ValidationError, match="ComponentsSettings"):
        Components(settings=_DummyResolverSettings())


def test_components_compute_clusters_deduplicates_resolver_assignments() -> None:
    assignments = Components(
        settings=ComponentsSettings(thresholds={}),
    ).compute_clusters(
        model_edges={},
        resolver_assignments={
            "resolver_a": pl.DataFrame(
                {
                    "parent_id": [10, 10, 10],
                    "child_id": [4, 3, 3],
                },
                schema={"parent_id": pl.UInt64, "child_id": pl.UInt64},
            )
        },
    )

    expected = pl.DataFrame(
        {
            "parent_id": [1, 1],
            "child_id": [3, 4],
        },
        schema={"parent_id": pl.UInt64, "child_id": pl.UInt64},
    )
    assert_frame_equal(assignments, expected)
