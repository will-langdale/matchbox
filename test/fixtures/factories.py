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
            strict=False,
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


def _calculate_max_possible_edges(n_nodes: int, num_components: int) -> int:
    """
    Calculate the max possible number of edges given n nodes split into k components.

    Args:
        n_nodes: Total number of nodes
        num_components: Number of components to split into

    Returns:
        Maximum possible number of edges
    """
    nodes_per_component = n_nodes // num_components
    max_edges_per_component = (
        nodes_per_component * nodes_per_component
    )  # Complete bipartite graph
    return max_edges_per_component * num_components


def _split_values_into_components(
    values: list[int], num_components: int
) -> list[np.ndarray]:
    """
    Split values into non-overlapping groups for each component.

    Args:
        values: List of values to split
        num_components: Number of components to create

    Returns:
        List of arrays, one for each component
    """
    values = np.array(values)
    np.random.shuffle(values)
    return np.array_split(values, num_components)


def generate_dummy_probabilities(
    left_values: list[int],
    right_values: list[int],
    prob_range: tuple[float, float],
    num_components: int,
    total_rows: int,
) -> pa.Table:
    """
    Generate dummy Arrow probabilities data with guaranteed isolated components.

    Args:
        left_values: List of integers to use for left column
        right_values: List of integers to use for right column
        prob_range: Tuple of (min_prob, max_prob) to constrain probabilities
        num_components: Number of distinct connected components to generate
        total_rows: Total number of rows to generate

    Returns:
        PyArrow Table with 'left', 'right', and 'probability' columns
    """
    # Validate inputs
    if len(left_values) < 2 or len(right_values) < 2:
        raise ValueError("Need at least 2 possible values for both left and right")
    if num_components > min(len(left_values), len(right_values)):
        raise ValueError(
            "Cannot have more components than minimum of left/right values"
        )

    min_nodes = min(len(left_values), len(right_values))
    max_possible_edges = _calculate_max_possible_edges(min_nodes, num_components)

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
    left_components = _split_values_into_components(left_values, num_components)
    right_components = _split_values_into_components(right_values, num_components)

    # Calculate base number of edges per component
    base_edges_per_component = total_rows // num_components
    remaining_edges = total_rows % num_components

    all_edges = []

    # Generate edges for each component
    for comp_idx in range(num_components):
        comp_left_values = left_components[comp_idx]
        comp_right_values = right_components[comp_idx]

        # Calculate edges for this component
        edges_in_component = base_edges_per_component
        if comp_idx < remaining_edges:  # Distribute remaining edges
            edges_in_component += 1

        # Ensure basic connectivity within the component
        base_edges = []

        # Create a spanning tree-like structure
        for i in range(len(comp_left_values)):
            base_edges.append(
                (
                    comp_left_values[i],
                    comp_right_values[i % len(comp_right_values)],
                    np.random.randint(prob_min, prob_max + 1),
                )
            )

        # Generate remaining random edges strictly within this component
        remaining_edges = edges_in_component - len(base_edges)
        if remaining_edges > 0:
            random_lefts = np.random.choice(comp_left_values, size=remaining_edges)
            random_rights = np.random.choice(comp_right_values, size=remaining_edges)
            random_probs = np.random.randint(
                prob_min, prob_max + 1, size=remaining_edges
            )

            component_edges = base_edges + list(
                zip(random_lefts, random_rights, random_probs, strict=False)
            )
        else:
            component_edges = base_edges

        all_edges.extend(component_edges)

    # Convert to arrays
    lefts, rights, probs = zip(*all_edges, strict=False)

    # Create PyArrow arrays
    left_array = pa.array(lefts, type=pa.uint64())
    right_array = pa.array(rights, type=pa.uint64())
    prob_array = pa.array(probs, type=pa.uint8())

    return pa.table(
        [left_array, right_array, prob_array], names=["left", "right", "probability"]
    )
