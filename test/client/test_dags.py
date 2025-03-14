from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Engine

from matchbox.client.dags import Dag, DedupeStep, LinkStep, StepInput
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.sources import source_factory


@patch("matchbox.client.dags._handler.index")
@patch.object(DedupeStep, "run")
@patch.object(LinkStep, "run")
def test_dag_runs(
    link_run: Mock, dedupe_run: Mock, handler_index: Mock, sqlite_warehouse: Engine
):
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
    s_foo, s_bar, s_d_foo, s_d_bar, s_foo_bar = (
        dag.sequence.index("foo"),
        dag.sequence.index("bar"),
        dag.sequence.index("d_foo"),
        dag.sequence.index("d_bar"),
        dag.sequence.index("foo_bar"),
    )
    assert s_foo < s_d_foo < s_foo_bar
    assert s_bar < s_d_bar < s_foo_bar

    # Run DAG
    dag.run()

    assert handler_index.call_count == 2
    assert {
        handler_index.call_args_list[0].kwargs["source"].address.full_name,
        handler_index.call_args_list[1].kwargs["source"].address.full_name,
    } == {"foo", "bar"}

    assert dedupe_run.call_count == 2
    dedupe_run.assert_called_with(engine=sqlite_warehouse)

    link_run.assert_called_once_with(engine=sqlite_warehouse)


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
