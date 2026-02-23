import pytest

from matchbox.common.dtos import (
    Collection,
    Match,
    QueryConfig,
    ResolutionPath,
)


def test_match_validates() -> None:
    """Match objects are validated when they're instantiated."""
    source_path = ResolutionPath(name="source", collection="default", run=1)
    target_path = ResolutionPath(name="target", collection="default", run=1)
    Match(
        cluster=1,
        source=source_path,
        source_id={"a"},
        target=target_path,
        target_id={"b"},
    )

    # Missing source_id with target_id
    with pytest.raises(ValueError):
        Match(cluster=1, source=source_path, target=target_path, target_id={"b"})

    # Missing cluster with target_id
    with pytest.raises(ValueError):
        Match(source=source_path, source_id={"a"}, target=target_path, target_id={"b"})

    # Missing source_id with cluster
    with pytest.raises(ValueError):
        Match(cluster=1, source=source_path, target=target_path)


def test_validate_collection() -> None:
    """Default run name needs to be within all runs."""
    # No default
    Collection(runs=[1, 2])

    # Valid default
    Collection(default_run=2, runs=[1, 2])

    with pytest.raises(ValueError):
        Collection(default_run=2, runs=[1])


def test_query_config_normalises_model_point_of_truth() -> None:
    """Model point-of-truth normalises to the canonical resolver name."""
    config = QueryConfig(
        source_resolutions=("source_a",),
        model_resolution="my_model",
    )

    assert config.model_resolution == "my_model"
    assert config.resolver_resolution == "resolver_my_model"
    assert config.point_of_truth == "resolver_my_model"


def test_query_config_rejects_inconsistent_resolution_pair() -> None:
    """Model and resolver fields must represent the same canonical pair."""
    with pytest.raises(ValueError, match="canonical"):
        QueryConfig(
            source_resolutions=("source_a",),
            model_resolution="my_model",
            resolver_resolution="resolver_other_model",
        )
