import pytest

from matchbox.common.dtos import (
    Collection,
    CollectionName,
    Match,
    ModelResolutionName,
    QueryConfig,
    ResolutionName,
    ResolutionPath,
    SourceResolutionName,
)
from matchbox.common.exceptions import MatchboxNameError


def test_validate_names():
    """Names are validated when they're instantiated."""
    name_classes = [
        CollectionName,
        ModelResolutionName,
        ResolutionName,
        SourceResolutionName,
    ]

    [NameClass("Valid.name_-") for NameClass in name_classes]

    for NameClass in name_classes:
        with pytest.raises(MatchboxNameError):
            NameClass("invalid name")


def test_match_validates():
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


def test_validate_collection():
    """Default run name needs to be within all runs."""
    # No default
    Collection(runs=[1, 2])

    # Valid default
    Collection(default_run=2, runs=[1, 2])

    with pytest.raises(ValueError):
        Collection(default_run=2, runs=[1])


def test_validate_query_paths():
    with pytest.raises(ValueError, match="Incompatible collection"):
        QueryConfig(
            source_resolutions=(
                ResolutionPath(collection="companies", run=1, name="source"),
            ),
            model_resolution=ResolutionPath(
                collection="companies", run=2, name="model"
            ),
        )

    with pytest.raises(ValueError, match="Incompatible collection"):
        QueryConfig(
            source_resolutions=(
                ResolutionPath(collection="companies", run=1, name="source"),
                ResolutionPath(collection="companies2", run=1, name="source2"),
            ),
            model_resolution=ResolutionPath(
                collection="companies", run=1, name="model"
            ),
        )
