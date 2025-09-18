from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Engine

from matchbox.client.dags import DAG
from matchbox.client.models import Model
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.client.sources import Source
from matchbox.common.factories.sources import source_factory


@patch.object(Source, "run")
@patch.object(Model, "run")
@patch.object(Source, "sync")
@patch.object(Model, "sync")
def test_dag_run_and_sync(
    model_sync_mock: Mock,
    source_sync_mock: Mock,
    model_run_mock: Mock,
    source_run_mock: Mock,
    sqlite_warehouse: Engine,
):
    """A legal DAG can be built and run."""
    # Set up constituents
    foo_tkit = source_factory(name="foo", engine=sqlite_warehouse).write_to_location()
    bar_tkit = source_factory(name="bar", engine=sqlite_warehouse).write_to_location()
    baz_tkit = source_factory(name="baz", engine=sqlite_warehouse).write_to_location()

    dag = DAG(collection_name="default")

    # Structure: sources can be added
    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())
    baz = dag.source(**baz_tkit.into_dag())

    assert set(dag.nodes.keys()) == {foo.name, bar.name, baz.name}

    # Structure: sources can be deduped
    d_foo = foo.query().deduper(
        name="d_foo",
        model_class=NaiveDeduper,
        model_settings={"id": "id", "unique_fields": []},
    )

    # Structure:
    # - sources can be passed directly to linkers
    # - or, linkers can take dedupers
    foo_bar = d_foo.query(foo).linker(
        bar.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"left_id": "id", "right_id": "id", "comparisons": ""},
    )

    # Structure: linkers can take other linkers
    foo_bar_baz = foo_bar.query(foo, bar, baz).linker(
        baz.query(),
        name="foo_bar_baz",
        model_class=DeterministicLinker,
        model_settings={"left_id": "id", "right_id": "id", "comparisons": ""},
    )

    assert set(dag.nodes.keys()) == {
        foo.name,
        bar.name,
        baz.name,
        d_foo.name,
        foo_bar.name,
        foo_bar_baz.name,
    }

    # Run DAG
    dag.run_and_sync()

    assert source_run_mock.call_count == 3
    assert source_sync_mock.call_count == 3
    assert model_run_mock.call_count == 3
    assert model_sync_mock.call_count == 3


def test_dags_missing_dependency(sqlite_warehouse: Engine):
    """Steps cannot be added before their dependencies."""
    dag = DAG("collection")
    foo = source_factory(name="foo", engine=sqlite_warehouse, dag=dag).source

    with pytest.raises(ValueError, match="not added to DAG"):
        dag.model(
            left_query=foo.query(),
            name="d_foo",
            model_class=NaiveDeduper,
            model_settings={"id": "id", "unique_fields": []},
        )

    # Failure leads to no dags being added
    assert not len(dag.nodes)
    assert not len(dag.graph)


def test_mixing_dags_fails(sqlite_warehouse: Engine):
    """Cannot reference a different DAG when adding a step."""
    dag = DAG("collection")
    dag2 = DAG("collection")

    foo_tkit = source_factory(name="foo", engine=sqlite_warehouse)
    foo = dag.source(**foo_tkit.into_dag())

    with pytest.raises(ValueError, match="mix DAGs"):
        # Different DAG in input
        dag2.model(
            left_query=foo.query(),
            name="d_foo",
            model_class=NaiveDeduper,
            model_settings={"id": "id", "unique_fields": []},
        )

    # Failure leads to no dags being added
    assert not len(dag2.nodes)
    assert not len(dag2.graph)


def test_dag_name_clash(sqlite_warehouse: Engine):
    """Names across sources and steps must be unique."""
    dag = DAG("collection")

    foo_tkit = source_factory(name="foo", engine=sqlite_warehouse)
    bar_tkit = source_factory(name="bar", engine=sqlite_warehouse)

    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())

    d_foo = foo.query().deduper(
        name="d_foo",
        model_class=NaiveDeduper,
        model_settings={"id": "id", "unique_fields": []},
    )

    with pytest.raises(ValueError, match="already taken"):
        bar.query().deduper(
            name="d_foo",
            model_class=NaiveDeduper,
            model_settings={"id": "id", "unique_fields": []},
        )

    # DAG is not modified by failed attempt
    assert dag.nodes["d_foo"] == d_foo
    # We didn't overwrite d_foo's dependencies
    assert dag.graph["d_foo"] == [foo.name]


def test_dag_disconnected(sqlite_warehouse: Engine):
    """Nodes cannot be disconnected."""
    dag = DAG("collection")

    foo_tkit = source_factory(name="foo", engine=sqlite_warehouse)
    bar_tkit = source_factory(name="bar", engine=sqlite_warehouse)

    dag.source(**foo_tkit.into_dag())
    dag.source(**bar_tkit.into_dag())

    with pytest.raises(ValueError, match="disconnected"):
        dag.run_and_sync()


def test_dag_draw(sqlite_warehouse: Engine):
    """Test that the draw method produces a correct string representation of the DAG."""
    # Set up a simple DAG
    foo_tkit = source_factory(name="foo", engine=sqlite_warehouse).write_to_location()
    bar_tkit = source_factory(name="bar", engine=sqlite_warehouse).write_to_location()
    baz_tkit = source_factory(name="baz", engine=sqlite_warehouse).write_to_location()

    dag = DAG(collection_name="default")

    # Structure: sources can be added
    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())
    baz = dag.source(**baz_tkit.into_dag())

    d_foo = foo.query().deduper(
        name="d_foo",
        model_class=NaiveDeduper,
        model_settings={"id": "id", "unique_fields": []},
    )

    foo_bar = d_foo.query(foo).linker(
        bar.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"left_id": "id", "right_id": "id", "comparisons": ""},
    )

    # Structure: linkers can take other linkers
    foo_bar_baz = foo_bar.query(foo, bar, baz).linker(
        baz.query(),
        name="foo_bar_baz",
        model_class=DeterministicLinker,
        model_settings={"left_id": "id", "right_id": "id", "comparisons": ""},
    )

    # Prepare the DAG and draw it

    # Test 1: Drawing without timestamps (original behavior)
    tree_str = dag.draw()

    # Verify the structure
    lines = tree_str.strip().split("\n")

    # The root node should be first
    assert lines[0] == "foo_bar_baz"

    # Check that all nodes are present
    node_names = [
        foo.name,
        bar.name,
        baz.name,
        d_foo.name,
        foo_bar.name,
        foo_bar_baz.name,
    ]

    for node in node_names:
        # Either the node name is at the start of a line or after the tree characters
        node_present = any(line.endswith(node) for line in lines)
        assert node_present, f"Node {node} not found in the tree representation"

    # Check that tree has correct formatting with tree characters
    tree_chars = ["└──", "├──", "│"]
    has_tree_chars = any(char in tree_str for char in tree_chars)
    assert has_tree_chars, (
        "Tree representation doesn't use expected formatting characters"
    )

    # Test 2: Drawing with timestamps (status indicators)
    # Set d_foo as processing and foo_bar as completed
    start_time = datetime.now()
    doing = "d_foo"
    foo_bar.last_run = datetime.now()

    # Draw the DAG with status indicators
    tree_str_with_status = dag.draw(start_time=start_time, doing=doing)
    status_lines = tree_str_with_status.strip().split("\n")

    # Verify status indicators are present
    status_indicators = ["✅", "🔄", "⏸️"]
    assert any(indicator in tree_str_with_status for indicator in status_indicators)

    # Check specific statuses: foo_bar done, d_foo working, others awaiting
    for line in status_lines:
        name = line.split()[-1]
        if name == "foo_bar":
            assert "✅" in line
        elif name == "d_foo":
            assert "🔄" in line
        elif name in [foo.name, bar.name, baz.name]:
            assert "⏸️" in line

    # Test 3: Check that node names are still present with status indicators
    for node in node_names:
        node_present = any(node in line for line in status_lines)
        assert node_present, (
            f"Node {node} not found in the tree representation with status indicators"
        )

    # Test 4: Drawing with skipped nodes
    skipped_nodes = [foo.name, d_foo.name]
    tree_str_with_skipped = dag.draw(
        start_time=start_time, doing=doing, skipped=skipped_nodes
    )
    skipped_lines = tree_str_with_skipped.strip().split("\n")

    # Check that skipped nodes have the skipped indicator
    for line in skipped_lines:
        name = line.split()[-1]
        if any(name == skipped for skipped in skipped_nodes):
            assert "⏭️" in line

    # Test all status indicators together
    doing = "foo_bar_baz"
    tree_str_all_statuses = dag.draw(
        start_time=start_time, doing=doing, skipped=skipped_nodes
    )
    assert all(
        indicator in tree_str_all_statuses for indicator in ["✅", "🔄", "⏸️", "⏭️"]
    )
