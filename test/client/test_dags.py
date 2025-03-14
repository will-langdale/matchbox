from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Engine

from matchbox.client.dags import Dag, DedupeStep, LinkStep, StepInput
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.sources import source_factory


@patch.object(DedupeStep, "run")
@patch.object(LinkStep, "run")
def test_dag_runs(dedupe_run: Mock, link_run: Mock, sqlite_warehouse: Engine):
    """A legal DAG can be built and run"""
    # Set up constituents
    foo = source_factory(full_name="foo").source
    bar = source_factory(full_name="bar").source

    d_foo = DedupeStep(
        name="d_foo",
        description="",
        cleaners={},
        left=StepInput(origin=foo, select={foo.address.full_name: []}),
        model_class=NaiveDeduper,
        settings={},
    )

    d_bar = DedupeStep(
        name="d_bar",
        description="",
        cleaners={},
        left=StepInput(origin=bar, select={bar.address.full_name: []}),
        model_class=NaiveDeduper,
        settings={},
    )

    foo_bar = LinkStep(
        left=StepInput(
            origin=d_foo,
            select={foo.address.full_name: []},
            cleaners={},
        ),
        right=StepInput(
            origin=d_bar,
            select={bar.address.full_name: []},
            cleaners={},
        ),
        name="foo_bar",
        description="",
        model_class=DeterministicLinker,
        settings={},
    )

    # Assemble DAG
    dag = Dag(engine=sqlite_warehouse)

    dag.add_sources(foo, bar)
    assert set(dag.nodes.keys()) == {"foo", "bar"}

    dag.add_steps(d_foo, d_bar, foo_bar)
    assert set(dag.nodes.keys()) == {"foo", "bar", "d_foo", "d_bar", "foo_bar"}

    # Prepare DAG
    dag.prepare()
    assert dag.sequence in (
        ["d_foo", "d_bar", "foo_bar"],
        ["d_bar", "d_foo", "foo_bar"],
    )

    # Run DAG
    dag.run()
    d_foo.run.assert_called()
    d_bar.run.assert_called()
    foo_bar.run.assert_called()


def test_dag_missing_dependency(sqlite_warehouse: Engine):
    """Steps cannot be added before their dependencies"""
    foo = source_factory(full_name="foo").source
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        cleaners={},
        left=StepInput(origin=foo, select={foo.address.full_name: []}),
        model_class=NaiveDeduper,
        settings={},
    )

    dag = Dag(engine=sqlite_warehouse)
    with pytest.raises(ValueError, match="Dependency"):
        dag.add_steps(d_foo)


def test_dag_disconnected(sqlite_warehouse: Engine):
    """Nodes cannot be disconnected"""
    foo = source_factory(full_name="foo").source
    bar = source_factory(full_name="bar").source

    dag = Dag(engine=sqlite_warehouse)
    dag.add_sources(foo, bar)

    with pytest.raises(ValueError, match="disconnected"):
        dag.prepare()
