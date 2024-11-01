import pytest
import rustworkx as rx
from matchbox.common.results import (
    ClusterResults,
    ModelMetadata,
    ModelType,
    ProbabilityResults,
)
from matchbox.server.postgresql.utils.insert import _cluster_results_to_hierarchical
from pandas import DataFrame


@pytest.fixture
def model_metadata():
    return ModelMetadata(
        name="test_model",
        description="Test model metadata",
        type=ModelType.DEDUPER,
        left_source="left",
    )


def create_results(prob_data, cluster_data, metadata):
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
                "hash": ["AB", "BC", "CD"],
                "left_id": ["A", "B", "C"],
                "right_id": ["B", "C", "D"],
                "probability": [1.0, 1.0, 1.0],
            },
            {
                "parent": ["ABCD", "ABCD", "ABCD", "ABCD"],
                "child": ["A", "B", "C", "D"],
                "threshold": [1.0, 1.0, 1.0, 1.0],
            },
            {
                ("AB", "A", 1.0),
                ("AB", "B", 1.0),
                ("BC", "B", 1.0),
                ("BC", "C", 1.0),
                ("CD", "C", 1.0),
                ("CD", "D", 1.0),
                ("ABCD", "AB", 1.0),
                ("ABCD", "BC", 1.0),
                ("ABCD", "CD", 1.0),
            },
        ),
        # Test case 2: Asymmetric probability components
        (
            {
                "hash": ["a", "j", "k"],
                "left_id": ["w", "x", "y"],
                "right_id": ["x", "y", "z"],
                "probability": [0.9, 0.85, 0.8],
            },
            {
                "parent": ["a", "a", "b", "b", "b", "c", "c", "c", "c"],
                "child": ["w", "x", "w", "x", "y", "w", "x", "y", "z"],
                "threshold": [0.9, 0.9, 0.85, 0.85, 0.85, 0.8, 0.8, 0.8, 0.8],
            },
            {
                ("a", "w", 0.9),
                ("a", "x", 0.9),
                ("j", "x", 0.85),
                ("j", "y", 0.85),
                ("b", "a", 0.85),
                ("b", "j", 0.85),
                ("k", "y", 0.8),
                ("k", "z", 0.8),
                ("c", "b", 0.8),
                ("c", "k", 0.8),
            },
        ),
        # Test case 3: Empty input
        (
            {
                "hash": [],
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
                "hash": ["a"],
                "left_id": ["x"],
                "right_id": ["y"],
                "probability": [0.9],
            },
            {
                "parent": ["a", "a"],
                "child": ["x", "y"],
                "threshold": [0.9, 0.9],
            },
            {
                ("a", "x", 0.9),
                ("a", "y", 0.9),
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
    actual_relations = set((p, c, t) for p, c, t in hierarchy)

    assert actual_relations == expected_relations

    if actual_relations:  # Skip verification for empty case
        verify_hierarchy(hierarchy)
