"""Tests for resolver node behaviour and public client API contracts."""

from unittest.mock import patch

import polars as pl
import pytest
from sqlalchemy import Engine

from matchbox.client.models import Model
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.client.resolvers import Components, ComponentsSettings
from matchbox.common.exceptions import MatchboxResolutionTypeError
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.sources import source_factory


def test_model_resolver_single_input_creation(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """A model can create a resolver rooted at itself."""
    dag = TestkitDAG().dag
    source_testkit = source_factory(
        engine=sqla_sqlite_warehouse,
        name="source",
    ).write_to_location()
    source = dag.source(**source_testkit.into_dag())

    model = source.query().deduper(
        name="source_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )

    resolver = model.resolver(
        name="resolver",
        resolver_class=Components,
        resolver_settings={"thresholds": {model.name: 0.0}},
    )

    assert dag.get_resolver(resolver.name) == resolver
    assert resolver.inputs == (model,)
    assert resolver.config.inputs == (model.name,)
    assert dag.graph[resolver.name] == [model.name]


def test_model_resolver_multiple_inputs_creation(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """A resolver created from a model keeps input order and config."""
    dag = TestkitDAG().dag
    source_testkit = source_factory(
        engine=sqla_sqlite_warehouse,
        name="source",
    ).write_to_location()
    target_testkit = source_factory(
        engine=sqla_sqlite_warehouse,
        name="target",
    ).write_to_location()

    source = dag.source(**source_testkit.into_dag())
    target = dag.source(**target_testkit.into_dag())

    linker = source.query().linker(
        target.query(),
        name="source_target_linker",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    source_dedupe = source.query().deduper(
        name="source_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )

    resolver = linker.resolver(
        source_dedupe,
        name="root_resolver",
        resolver_class=Components,
        resolver_settings=ComponentsSettings(
            thresholds={linker.name: 0.0, source_dedupe.name: 0.0}
        ),
    )

    assert resolver.inputs == (linker, source_dedupe)
    assert resolver.config.inputs == (linker.name, source_dedupe.name)
    assert dag.graph[resolver.name] == [linker.name, source_dedupe.name]


def test_model_resolver_rejects_resolver_input(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Resolver input type checks are preserved through Model.resolver."""
    dag = TestkitDAG().dag
    source_testkit = source_factory(
        engine=sqla_sqlite_warehouse,
        name="source",
    ).write_to_location()
    source = dag.source(**source_testkit.into_dag())

    dedupe = source.query().deduper(
        name="source_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )
    first_resolver = dedupe.resolver(
        name="resolver_1",
        resolver_class=Components,
        resolver_settings={"thresholds": {dedupe.name: 0.0}},
    )

    with pytest.raises(MatchboxResolutionTypeError, match="Expected one of: model"):
        dedupe.resolver(
            first_resolver,
            name="resolver_2",
            resolver_class=Components,
            resolver_settings={
                "thresholds": {first_resolver.name: 0.0, dedupe.name: 0.0}
            },
        )


def test_resolver_run_requires_materialised_inputs(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Resolver run requires upstream model results to exist locally."""
    dag = TestkitDAG().dag
    source_testkit = source_factory(
        engine=sqla_sqlite_warehouse,
        name="source",
    ).write_to_location()
    source = dag.source(**source_testkit.into_dag())

    model = source.query().deduper(
        name="source_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )
    resolver = model.resolver(
        name="resolver",
        resolver_class=Components,
        resolver_settings={"thresholds": {model.name: 0.0}},
    )

    with pytest.raises(ValueError, match="has no local results"):
        resolver.run()


def test_resolver_prepare_downloads_only_missing_inputs(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Resolver prepare only downloads dependencies with missing local data."""
    dag = TestkitDAG().dag
    source_testkit = source_factory(
        engine=sqla_sqlite_warehouse,
        name="source",
    ).write_to_location()
    target_testkit = source_factory(
        engine=sqla_sqlite_warehouse,
        name="target",
    ).write_to_location()
    source = dag.source(**source_testkit.into_dag())
    target = dag.source(**target_testkit.into_dag())

    linker = source.query().linker(
        target.query(),
        name="source_target_linker",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    source_dedupe = source.query().deduper(
        name="source_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )

    resolver = linker.resolver(
        source_dedupe,
        name="root_resolver",
        resolver_class=Components,
        resolver_settings=ComponentsSettings(
            thresholds={linker.name: 0.0, source_dedupe.name: 0.0}
        ),
    )

    source_dedupe.results = pl.DataFrame(
        {"left_id": ["1"], "right_id": ["1"], "probability": [1.0]}
    )

    with patch.object(Model, "download", autospec=True) as download_mock:
        resolver.prepare()

    assert download_mock.call_count == 1
    assert download_mock.call_args.args[0] is linker
