from typing import Iterable

import pytest
import rustworkx as rx
from pandas import DataFrame
from sqlalchemy import text

from matchbox.client.results import (
    ClusterResults,
    ModelMetadata,
    ModelType,
    ProbabilityResults,
)
from matchbox.server.postgresql.benchmark.generate_tables import generate_all_tables
from matchbox.server.postgresql.benchmark.init_schema import create_tables, empty_schema
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.utils.insert import _cluster_results_to_hierarchical


@pytest.fixture
def model_metadata():
    return ModelMetadata(
        name="test_model",
        description="Test model metadata",
        type=ModelType.DEDUPER,
        left_source="left",
    )


def create_results(prob_data: dict, cluster_data: dict, metadata: ModelMetadata):
    """Helper to create ProbabilityResults and ClusterResults from test data."""
    prob_df = DataFrame(prob_data)
    cluster_df = DataFrame(cluster_data)

    return (
        ProbabilityResults(
            dataframe=prob_df.convert_dtypes(dtype_backend="pyarrow"),
            metadata=metadata,
        ),
        ClusterResults(
            dataframe=cluster_df.convert_dtypes(dtype_backend="pyarrow"),
            metadata=metadata,
        ),
    )


def verify_hierarchy(hierarchy: list[tuple[bytes, bytes, float]]) -> None:
    """
    Verify each item has exactly one ultimate parent at each relevant threshold.

    Args:
        hierarchy: List of (parent, child, threshold) relationships
    """
    # Group relationships by threshold
    thresholds = sorted({t for _, _, t in hierarchy}, reverse=True)

    for threshold in thresholds:
        # Build graph of relationships at this threshold
        graph = rx.PyDiGraph()
        nodes = {}  # hash -> node_id

        # Add all nodes first
        edges = [(p, c) for p, c, t in hierarchy if t >= threshold]
        items = set()  # Track individual items (leaves)

        for parent, child in edges:
            if parent not in nodes:
                nodes[parent] = graph.add_node(parent)
            if child not in nodes:
                nodes[child] = graph.add_node(child)
            # If this child never appears as a parent, it's an item
            if child not in {p for p, _ in edges}:
                items.add(child)

        # Add edges
        for parent, child in edges:
            graph.add_edge(nodes[parent], nodes[child], None)

        # For each item, find its ultimate parents
        for item in items:
            item_node = nodes[item]
            ancestors = set()

            # Find all ancestors that have no parents themselves
            for node in graph.node_indices():
                node_hash = graph.get_node_data(node)
                if (
                    rx.has_path(graph, node, item_node) or node == item_node
                ) and graph.in_degree(node) == 0:
                    ancestors.add(node_hash)

            assert len(ancestors) == 1, (
                f"Item {item} has {len(ancestors)} ultimate parents at "
                f"threshold {threshold}: {ancestors}"
            )


@pytest.mark.parametrize(
    ("prob_data", "cluster_data", "expected_relations"),
    [
        # Test case 1: Equal probability components
        (
            {
                "id": ["ab", "bc", "cd"],
                "left_id": ["a", "b", "c"],
                "right_id": ["b", "c", "d"],
                "probability": [1.0, 1.0, 1.0],
            },
            {
                "parent": ["abcd", "abcd", "abcd", "abcd"],
                "child": ["a", "b", "c", "d"],
                "threshold": [1.0, 1.0, 1.0, 1.0],
            },
            {
                ("ab", "a", 1.0),
                ("ab", "b", 1.0),
                ("bc", "b", 1.0),
                ("bc", "c", 1.0),
                ("cd", "c", 1.0),
                ("cd", "d", 1.0),
                ("abcd", "ab", 1.0),
                ("abcd", "bc", 1.0),
                ("abcd", "cd", 1.0),
            },
        ),
        # Test case 2: Asymmetric probability components
        (
            {
                "id": ["wx", "xy", "yz"],
                "left_id": ["w", "x", "y"],
                "right_id": ["x", "y", "z"],
                "probability": [0.9, 0.85, 0.8],
            },
            {
                "parent": [
                    "wx",
                    "wx",
                    "wxy",
                    "wxy",
                    "wxy",
                    "wxyz",
                    "wxyz",
                    "wxyz",
                    "wxyz",
                ],
                "child": ["w", "x", "w", "x", "y", "w", "x", "y", "z"],
                "threshold": [0.9, 0.9, 0.85, 0.85, 0.85, 0.8, 0.8, 0.8, 0.8],
            },
            {
                ("wx", "w", 0.9),
                ("wx", "x", 0.9),
                ("xy", "x", 0.85),
                ("xy", "y", 0.85),
                ("wxy", "wx", 0.85),
                ("wxy", "xy", 0.85),
                ("yz", "y", 0.8),
                ("yz", "z", 0.8),
                ("wxyz", "wxy", 0.8),
                ("wxyz", "yz", 0.8),
            },
        ),
        # Test case 3: Empty input
        (
            {
                "id": [],
                "left_id": [],
                "right_id": [],
                "probability": [],
            },
            {
                "parent": [],
                "child": [],
                "threshold": [],
            },
            set(),
        ),
        # Test case 4: Single two-item component
        (
            {
                "id": ["xy"],
                "left_id": ["x"],
                "right_id": ["y"],
                "probability": [0.9],
            },
            {
                "parent": ["xy", "xy"],
                "child": ["x", "y"],
                "threshold": [0.9, 0.9],
            },
            {
                ("xy", "x", 0.9),
                ("xy", "y", 0.9),
            },
        ),
    ],
    ids=["equal_prob", "asymmetric_prob", "empty", "single_component"],
)
def test_cluster_results_to_hierarchical(
    prob_data, cluster_data, expected_relations, model_metadata
):
    """Test hierarchical clustering with various input scenarios."""
    prob_results, cluster_results = create_results(
        prob_data, cluster_data, model_metadata
    )

    hierarchy = _cluster_results_to_hierarchical(prob_results, cluster_results)
    actual_relations = set((p, c, t) for p, c, t in hierarchy.itertuples(index=False))

    assert actual_relations == expected_relations

    if actual_relations:  # Skip verification for empty case
        verify_hierarchy(hierarchy.itertuples(index=False))


def test_benchmark_init_schema():
    schema = MBDB.MatchboxBase.metadata.schema
    count_tables = text(f"""
        select count(*)
        from information_schema.tables
        where table_schema = '{schema}';
    """)

    with MBDB.get_engine().connect() as con:
        con.execute(text(empty_schema()))
        con.commit()
        n_tables = int(con.execute(count_tables).scalar())
        assert n_tables == 0

        con.execute(text(create_tables()))
        con.commit()
        n_tables_expected = len(MBDB.MatchboxBase.metadata.tables)
        n_tables = int(con.execute(count_tables).scalar())
        assert n_tables == n_tables_expected


def test_benchmark_generate_tables():
    schema = MBDB.MatchboxBase.metadata.schema

    def array_encode(array: Iterable[str]):
        if not array:
            return None
        escaped_l = [f'"{s}"' for s in array]
        list_rep = ", ".join(escaped_l)
        return "{" + list_rep + "}"

    with MBDB.get_engine().connect() as con:
        con.execute(text(empty_schema()))
        con.commit()

        results = generate_all_tables(20, 5, 25, 5, 25)

        assert len(results) == len(MBDB.MatchboxBase.metadata.tables)

        for table_name, table_arrow in results.items():
            df = table_arrow.to_pandas()
            # Pandas' `to_sql` dislikes arrays
            if "source_pk" in df.columns:
                df["source_pk"] = df["source_pk"].apply(array_encode)
            # Pandas' `to_sql` dislikes large unsigned ints
            for c in df.columns:
                if df[c].dtype == "uint64":
                    df[c] = df[c].astype("int64")
            df.to_sql(table_name, con, schema)
