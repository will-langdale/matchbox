from collections import Counter

import numpy as np
import pyarrow as pa
import rustworkx as rx


def verify_components(table) -> dict:
    """
    Fast verification of connected components using rustworkx.

    Args:
        table: PyArrow table with 'left', 'right' columns

    Returns:
        dictionary containing basic component statistics
    """
    graph = rx.PyGraph()

    unique_nodes = set(table["left"].to_numpy()) | set(table["right"].to_numpy())
    graph.add_nodes_from(range(len(unique_nodes)))

    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    edges = [
        (node_to_idx[left], node_to_idx[right])
        for left, right in zip(
            table["left"].to_numpy(),
            table["right"].to_numpy(),
            strict=True,
        )
    ]

    graph.add_edges_from_no_data(edges)

    components = rx.connected_components(graph)
    component_sizes = Counter(len(component) for component in components)

    return {
        "num_components": len(components),
        "total_nodes": len(unique_nodes),
        "total_edges": len(edges),
        "component_sizes": component_sizes,
        "min_component_size": min(component_sizes.keys()),
        "max_component_size": max(component_sizes.keys()),
    }


def _calculate_max_possible_edges(
    n_nodes: int, num_components: int, deduplicate: bool
) -> int:
    """
    Calculate a conservative max number of edges given n nodes split into k components.

    Args:
        n_nodes: Total number of nodes on smallest table (either left or right)
        num_components: Number of components to split into
        deduplicate: Whether we are dealing with probabilities for deduplication

    Returns:
        Maximum possible number of edges
    """
    # Size of smallest components we will generate. Because some components might be
    # larger, the final estimate might be smaller than necessary
    min_nodes_per_component = n_nodes // num_components
    if deduplicate:
        # Max edges in undirected graph of size n
        max_edges_per_component = (
            min_nodes_per_component * (min_nodes_per_component - 1) / 2
        )
    else:
        # Complete bipartite graph
        max_edges_per_component = min_nodes_per_component * min_nodes_per_component
    return max_edges_per_component * num_components


def generate_dummy_probabilities(
    left_values: list[int],
    right_values: list[int] | None,
    prob_range: tuple[float, float],
    num_components: int,
    total_rows: int,
) -> pa.Table:
    """
    Generate dummy Arrow probabilities data with guaranteed isolated components.

    Args:
        left_values: List of integers to use for left column
        right_values: List of integers to use for right column. If None, assume we
            are generating probabilities for deduplication
        prob_range: Tuple of (min_prob, max_prob) to constrain probabilities
        num_components: Number of distinct connected components to generate
        total_rows: Total number of rows to generate

    Returns:
        PyArrow Table with 'left', 'right', and 'probability' columns
    """
    # Validate inputs
    deduplicate = False
    if right_values is None:
        right_values = left_values
        deduplicate = True

    if len(left_values) < 2 or len(right_values) < 2:
        raise ValueError("Need at least 2 possible values for both left and right")
    if num_components > min(len(left_values), len(right_values)):
        raise ValueError(
            "Cannot have more components than minimum of left/right values"
        )

    min_nodes = min(len(left_values), len(right_values))
    max_possible_edges = _calculate_max_possible_edges(
        min_nodes, num_components, deduplicate
    )

    if total_rows > max_possible_edges:
        raise ValueError(
            f"Cannot generate {total_rows:,} edges with {num_components:,} components. "
            f"Max possible edges is {max_possible_edges:,} given {min_nodes:,} nodes. "
            "Either increase the number of nodes, decrease the number of components, "
            "or decrease the total edges requested."
        )

    # Convert probability range to integers (60-80 for 0.60-0.80)
    prob_min = int(prob_range[0] * 100)
    prob_max = int(prob_range[1] * 100)

    # Split values into completely separate groups for each component
    left_components = np.array_split(np.array(left_values), num_components)
    right_components = np.array_split(np.array(right_values), num_components)
    # For each left-right component pair, the right equals the left rotated by one
    right_components = [np.roll(c, 1) for c in right_components]

    # Calculate base number of edges per component
    base_edges_per_component = total_rows // num_components
    remaining_edges = total_rows % num_components

    all_edges = []

    # Generate edges for each component
    for comp_idx in range(num_components):
        comp_left_values = left_components[comp_idx]
        comp_right_values = right_components[comp_idx]

        edges_in_component = base_edges_per_component
        # Distribute remaining edges, one per component
        if comp_idx < remaining_edges:
            edges_in_component += 1

        # Ensure basic connectivity within the component by creating a spanning-tree
        # like structure
        base_edges = set()
        for i in range(len(comp_left_values)):
            left = comp_left_values[i]
            right = comp_right_values[i % len(comp_right_values)]
            if deduplicate:
                left, right = sorted([left, right])

            base_edges.add((left, right))

        # Remove self-references from base edges
        base_edges = {(le, ri) for le, ri in base_edges if le != ri}

        # Generate remaining random edges strictly within this component
        # TODO: this can certainly be optimised
        all_possible_edges = [
            (x, y)
            for x in comp_left_values
            for y in comp_right_values
            if x != y and (x, y) not in base_edges
        ]
        edges_required = edges_in_component - len(base_edges)
        extra_edges_idx = np.random.choice(
            len(all_possible_edges), size=edges_required, replace=False
        )
        extra_edges = [
            e for i, e in enumerate(all_possible_edges) if i in extra_edges_idx
        ]
        component_edges = list(base_edges) + extra_edges
        random_probs = np.random.randint(
            prob_min, prob_max + 1, size=len(component_edges)
        )

        component_edges = [
            (le, ri, pr)
            for (le, ri), pr in zip(component_edges, random_probs, strict=True)
        ]

        all_edges.extend(component_edges)

    # Convert to arrays
    lefts, rights, probs = zip(*all_edges, strict=True)

    # Create PyArrow arrays
    left_array = pa.array(lefts, type=pa.uint64())
    right_array = pa.array(rights, type=pa.uint64())
    prob_array = pa.array(probs, type=pa.uint8())

    return pa.table(
        [left_array, right_array, prob_array], names=["left", "right", "probability"]
    )
