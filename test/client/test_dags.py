from unittest.mock import Mock, call, patch

import pytest
from sqlalchemy import Engine

from matchbox.client.dags import DAG, DedupeStep, LinkStep, StepInput
from matchbox.client.helpers.selector import Selector
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.sources import source_factory


def test_dedupe_step_run(
    sqlite_warehouse: Engine,
):
    """Tests that a dedupe step orchestrates lower-level API correctly."""
    with (
        patch("matchbox.client.dags.make_model") as make_model_mock,
        patch("matchbox.client.dags.process") as process_mock,
        patch("matchbox.client.dags.query") as query_mock,
    ):
        # Complete mock set up
        model_mock = Mock()
        make_model_mock.return_value = model_mock

        results_mock = Mock()
        results_mock.to_matchbox = Mock()

        model_mock.run = Mock(return_value=results_mock)

        # Set up and run deduper
        foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
        d_foo = DedupeStep(
            name="d_foo",
            description="",
            left=StepInput(prev_node=foo, select={foo: []}, threshold=0.5),
            model_class=NaiveDeduper,
            settings={"id": "id", "unique_fields": []},
        )

        d_foo.run()

        # Right data is queried
        query_mock.assert_called_once_with(
            [Selector(source=foo, fields=[])],
            return_type="pandas",
            threshold=d_foo.left.threshold,
            resolution_name=d_foo.left.name,
        )
        # Data is pre-processed
        process_mock.assert_called_once_with(
            query_mock.return_value, d_foo.left.cleaners
        )

        # Model is created and run
        make_model_mock.assert_called_once_with(
            model_name=d_foo.name,
            description=d_foo.description,
            model_class=d_foo.model_class,
            model_settings=d_foo.settings,
            left_data=process_mock.return_value,
            left_resolution=d_foo.left.name,
        )
        model_mock.run.assert_called_once()

        # Results are stored
        model_mock.run().to_matchbox.assert_called_once()


def test_link_step_run(
    sqlite_warehouse: Engine,
):
    """Tests that a link step orchestrates lower-level API correctly."""
    with (
        patch("matchbox.client.dags.make_model") as make_model_mock,
        patch("matchbox.client.dags.process") as process_mock,
        patch("matchbox.client.dags.query") as query_mock,
    ):
        # Complete mock set up
        model_mock = Mock()
        make_model_mock.return_value = model_mock

        results_mock = Mock()
        results_mock.to_matchbox = Mock()

        model_mock.run = Mock(return_value=results_mock)

        # Set up and run deduper
        foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
        bar = source_factory(full_name="bar", engine=sqlite_warehouse).source
        foo_bar = LinkStep(
            name="foo_bar",
            description="",
            left=StepInput(prev_node=foo, select={foo: []}, threshold=0.5),
            right=StepInput(prev_node=bar, select={bar: []}, threshold=0.7),
            model_class=DeterministicLinker,
            settings={"left_id": "id", "right_id": "id", "comparisons": ""},
        )

        foo_bar.run()

        # Right data is queried
        assert query_mock.call_count == 2
        assert query_mock.call_args_list[0] == call(
            [Selector(source=foo, fields=[])],
            return_type="pandas",
            threshold=foo_bar.left.threshold,
            resolution_name=foo_bar.left.name,
        )
        assert query_mock.call_args_list[1] == call(
            [Selector(source=bar, fields=[])],
            return_type="pandas",
            threshold=foo_bar.right.threshold,
            resolution_name=foo_bar.right.name,
        )

        # Data is pre-processed
        assert process_mock.call_count == 2
        assert process_mock.call_args_list[0] == call(
            query_mock.return_value, foo_bar.left.cleaners
        )
        assert process_mock.call_args_list[1] == call(
            query_mock.return_value, foo_bar.right.cleaners
        )

        # Model is created and run
        make_model_mock.assert_called_once_with(
            model_name=foo_bar.name,
            description=foo_bar.description,
            model_class=foo_bar.model_class,
            model_settings=foo_bar.settings,
            left_data=process_mock.return_value,
            left_resolution=foo_bar.left.name,
            right_data=process_mock.return_value,
            right_resolution=foo_bar.right.name,
        )
        model_mock.run.assert_called_once()

        # Results are stored
        model_mock.run().to_matchbox.assert_called_once()


@patch("matchbox.client.dags._handler.index")
@patch.object(DedupeStep, "run")
@patch.object(LinkStep, "run")
def test_dag_runs(
    link_run: Mock, dedupe_run: Mock, handler_index: Mock, sqlite_warehouse: Engine
):
    """A legal DAG can be built and run."""
    # Set up constituents
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    bar = source_factory(full_name="bar", engine=sqlite_warehouse).source
    baz = source_factory(full_name="baz", engine=sqlite_warehouse).source

    # Structure: Sources can be deduped
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
    )

    # Structure:
    # - Sources can be passed directly to linkers
    # - Or, linkers can take dedupers
    foo_bar = LinkStep(
        left=StepInput(
            prev_node=d_foo,
            select={foo: []},
            cleaners={},
        ),
        right=StepInput(
            prev_node=bar,
            select={bar: []},
            cleaners={},
        ),
        name="foo_bar",
        description="",
        model_class=DeterministicLinker,
        settings={},
    )

    # Structure: Linkers can take other linkers
    foo_bar_baz = LinkStep(
        left=StepInput(
            prev_node=foo_bar,
            select={foo: [], bar: []},
            cleaners={},
        ),
        right=StepInput(
            prev_node=baz,
            select={baz: []},
            cleaners={},
        ),
        name="foo_bar_baz",
        description="",
        model_class=DeterministicLinker,
        settings={},
    )

    # Assemble DAG
    dag = DAG()

    dag.add_sources(foo, bar, baz)
    assert set(dag.nodes.keys()) == {
        str(foo.address),
        str(bar.address),
        str(baz.address),
    }

    dag.add_steps(d_foo, foo_bar, foo_bar_baz)
    assert set(dag.nodes.keys()) == {
        str(foo.address),
        str(bar.address),
        str(baz.address),
        d_foo.name,
        foo_bar.name,
        foo_bar_baz.name,
    }
    assert d_foo.sources == {foo.address}
    assert foo_bar.sources == {foo.address, bar.address}
    assert foo_bar_baz.sources == {foo.address, bar.address, baz.address}

    # Prepare DAG
    dag.prepare()
    s_foo, s_bar, s_d_foo, s_foo_bar, s_foo_bar_baz = (
        dag.sequence.index(str(foo.address)),
        dag.sequence.index(str(bar.address)),
        dag.sequence.index(d_foo.name),
        dag.sequence.index(foo_bar.name),
        dag.sequence.index(foo_bar_baz.name),
    )
    assert s_foo < s_d_foo < s_foo_bar < s_foo_bar_baz
    assert s_bar < s_foo_bar < s_foo_bar_baz

    # Run DAG
    dag.run()

    assert handler_index.call_count == 3
    assert {
        handler_index.call_args_list[0].kwargs["source"],
        handler_index.call_args_list[1].kwargs["source"],
        handler_index.call_args_list[2].kwargs["source"],
    } == {foo, bar, baz}

    assert dedupe_run.call_count == 1
    assert link_run.call_count == 2


def test_dag_missing_dependency(sqlite_warehouse: Engine):
    """Steps cannot be added before their dependencies."""
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
    )

    dag = DAG()
    with pytest.raises(ValueError, match="Dependency"):
        dag.add_steps(d_foo)
    # Sources are not added
    assert not d_foo.sources


def test_dag_name_clash(sqlite_warehouse: Engine):
    """Names across sources and steps must be unique."""
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    bar = source_factory(full_name="bar", engine=sqlite_warehouse).source
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
    )
    d_bar_wrong = DedupeStep(
        # Name clash!
        name="d_foo",
        description="",
        left=StepInput(prev_node=bar, select={bar: []}),
        model_class=NaiveDeduper,
        settings={},
    )
    dag = DAG()
    dag.add_sources(foo)
    dag.add_steps(d_foo)
    with pytest.raises(ValueError, match="already taken"):
        dag.add_steps(d_bar_wrong)
    # DAG is not modified by failed attempt
    assert dag.nodes["d_foo"] == d_foo
    # We didn't overwrite d_foo's dependencies
    assert dag.graph["d_foo"] == [str(foo.address)]
    # Sources are not added
    assert not d_bar_wrong.sources


def test_dag_source_unavailable(sqlite_warehouse: Engine):
    """Cannot select sources not available to a step."""
    # Set up all nodes
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    bar = source_factory(full_name="bar", engine=sqlite_warehouse).source

    d_foo_right = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
    )

    d_foo_wrong = DedupeStep(
        name="d_foo",
        description="",
        # Previous node and select disagree
        left=StepInput(prev_node=foo, select={bar: []}),
        model_class=NaiveDeduper,
        settings={},
    )

    bar_foo_wrong = LinkStep(
        name="foo_bar",
        description="",
        left=StepInput(prev_node=bar, select={bar: []}),
        # Previous node and select disagree
        right=StepInput(prev_node=d_foo_right, select={bar: []}),
        model_class=DeterministicLinker,
        settings={},
    )

    # CASE 1: Reading from Source
    dag = DAG()
    dag.add_sources(foo)
    with pytest.raises(ValueError, match="only select"):
        dag.add_steps(d_foo_wrong)
    # DAG is not modified by failed attempt
    assert d_foo_wrong.name not in dag.graph and d_foo_wrong.name not in dag.nodes

    # CASE 2: Reading from previous step
    dag = DAG()
    dag.add_sources(foo, bar)
    dag.add_steps(d_foo_right)
    with pytest.raises(ValueError, match="cannot select"):
        dag.add_steps(bar_foo_wrong)
    # DAG is not modified by failed attempt
    assert bar_foo_wrong.name not in dag.graph and bar_foo_wrong.name not in dag.nodes
    assert not bar_foo_wrong.sources


def test_dag_disconnected(sqlite_warehouse: Engine):
    """Nodes cannot be disconnected."""
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    bar = source_factory(full_name="bar", engine=sqlite_warehouse).source

    dag = DAG()
    dag.add_sources(foo, bar)

    with pytest.raises(ValueError, match="disconnected"):
        dag.prepare()
