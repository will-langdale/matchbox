import pytest

from matchbox.common.dtos import (
    Collection,
    CollectionName,
    Match,
    ModelResolutionName,
    ModelType,
    QueryConfig,
    ResolutionName,
    ResolutionPath,
    SourceResolutionName,
)
from matchbox.common.exceptions import MatchboxNameError
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import source_factory


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
    """Resolution paths are validated for compatibility in QueryConfig"""
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


def test_query_swappable():
    """Queries can be extended but not reduced."""
    # Can add sources (and models), but not remove them
    one_source = QueryConfig(
        source_resolutions=(
            ResolutionPath(collection="companies", run=1, name="source"),
        ),
    )

    two_sources = QueryConfig(
        source_resolutions=(
            ResolutionPath(collection="companies", run=1, name="source"),
            ResolutionPath(collection="companies", run=1, name="source2"),
        ),
        model_resolution=ResolutionPath(collection="companies", run=1, name="model"),
    )

    assert one_source.swappable_for(two_sources)
    assert not two_sources.swappable_for(one_source)

    # Can add cleaning outputs, but not remove them
    two_fields = QueryConfig(
        source_resolutions=(
            ResolutionPath(collection="companies", run=1, name="source"),
        ),
        cleaning={"a": "a", "b": "b"},
    )

    one_field = QueryConfig(
        source_resolutions=(
            ResolutionPath(collection="companies", run=1, name="source"),
        ),
        cleaning={"a": "different_a"},
    )

    assert one_field.swappable_for(two_fields)
    assert not two_fields.swappable_for(one_field)

    # If a query has cleaning, and one does not, they can't be swapped. Sorry.
    no_cleaning = QueryConfig(
        source_resolutions=(
            ResolutionPath(collection="companies", run=1, name="source"),
        )
    )

    assert not no_cleaning.swappable_for(one_field)
    assert not one_field.swappable_for(no_cleaning)


def test_source_swappable():
    """Sources can be extended but not reduced"""
    all_fields_config = source_factory().source.config
    some_fields_config = all_fields_config.model_copy(
        update={"index_fields": list(all_fields_config.index_fields)[1:]}
    )

    assert some_fields_config.swappable_for(all_fields_config)
    assert not all_fields_config.swappable_for(some_fields_config)


def test_model_swappable():
    """Models can be extended but not reduced"""
    link_config = model_factory(model_type=ModelType.LINKER).model.config
    dedupe_config = link_config.model_copy(update={"right_query": None})

    assert dedupe_config.swappable_for(link_config)
    assert not link_config.swappable_for(dedupe_config)
