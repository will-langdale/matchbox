from collections import Counter
from textwrap import dedent
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import rustworkx as rx
from faker import Faker
from pydantic import BaseModel, ConfigDict

from matchbox.common.dtos import ModelMetadata, ModelType
from matchbox.common.transform import graph_results


def verify_components(all_nodes: list[Any], table: pa.Table) -> dict:
    """Fast verification of connected components using rustworkx.

    Args:
        all_nodes: list of identities of inputs being matched
        table: PyArrow table with 'left', 'right' columns

    Returns:
        dictionary containing basic component statistics
    """
    graph, _, _ = graph_results(table, all_nodes)
    components = rx.connected_components(graph)
    component_sizes = Counter(len(component) for component in components)

    return {
        "num_components": len(components),
        "total_nodes": graph.num_nodes(),
        "total_edges": graph.num_edges(),
        "component_sizes": component_sizes,
        "min_component_size": min(component_sizes.keys()),
        "max_component_size": max(component_sizes.keys()),
    }


def _min_edges_component(left: int, right: int, deduplicate: bool) -> int:
    """Calculate min edges for component to be connected.

    Does so by assuming a spanning tree.

    Args:
        left: number of nodes of component on the left
        right: number of nodes of component on the right (for linking)
        deduplicate: whether edges are for deduplication

    Returns:
        Minimum number of edges
    """
    if not deduplicate:
        return left + right - 1

    return left - 1


def _max_edges_component(left: int, right: int, deduplicate: bool) -> int:
    """Calculate max edges for component to be avoid duplication.
    Considers complete graph for deduping, and complete bipartite graph for linking.

    Args:
        left: number of nodes of component on the left
        right: number of nodes of component on the right (for linking)
        deduplicate: whether edges are for deduplication

    Returns:
        Maximum number of edges
    """
    if not deduplicate:
        return left * right
    # n*(n-1) is always divisible by 2
    return left * (left - 1) // 2


def calculate_min_max_edges(
    left_nodes: int, right_nodes: int, num_components: int, deduplicate: bool
) -> tuple[int, int]:
    """Calculate min and max edges for a graph.

    Args:
        left_nodes: number of nodes in left source
        right_nodes: number of nodes in right source
        num_components: number of requested components
        deduplicate: whether edges are for deduplication

    Returns:
        Two-tuple representing min and max edges
    """
    left_mod, right_mod = left_nodes % num_components, right_nodes % num_components
    left_div, right_div = left_nodes // num_components, right_nodes // num_components

    min_mod, max_mod = sorted([left_mod, right_mod])

    min_edges, max_edges = 0, 0
    # components where both sides have maximum nodes
    min_edges += (
        _min_edges_component(left_div + 1, right_div + 1, deduplicate) * min_mod
    )
    max_edges += (
        _max_edges_component(left_div + 1, right_div + 1, deduplicate) * min_mod
    )
    # components where one side has maximum nodes
    left_after_min_mod, right_after_min_mod = left_div + 1, right_div
    if left_mod == min_mod:
        left_after_min_mod, right_after_min_mod = left_div, right_div + 1
    min_edges += _min_edges_component(
        left_after_min_mod, right_after_min_mod, deduplicate
    ) * (max_mod - min_mod)
    max_edges += _max_edges_component(
        left_after_min_mod, right_after_min_mod, deduplicate
    ) * (max_mod - min_mod)
    # components where both side have minimum nodes
    min_edges += _min_edges_component(left_div, right_div, deduplicate) * (
        num_components - max_mod
    )
    max_edges += _max_edges_component(left_div, right_div, deduplicate) * (
        num_components - max_mod
    )

    return min_edges, max_edges


def generate_dummy_probabilities(
    left_values: list[int],
    right_values: list[int] | None,
    prob_range: tuple[float, float],
    num_components: int,
    total_rows: int | None = None,
    seed: int = 42,
) -> pa.Table:
    """Generate dummy Arrow probabilities data with guaranteed isolated components.

    Args:
        left_values: List of integers to use for left column
        right_values: List of integers to use for right column. If None, assume we
            are generating probabilities for deduplication
        prob_range: Tuple of (min_prob, max_prob) to constrain probabilities
        num_components: Number of distinct connected components to generate
        total_rows: Total number of rows to generate

    Returns:
        PyArrow Table with 'left_id', 'right_id', and 'probability' columns
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

    left_nodes, right_nodes = len(left_values), len(right_values)
    min_possible_edges, max_possible_edges = calculate_min_max_edges(
        left_nodes, right_nodes, num_components, deduplicate
    )

    mode = "dedupe" if deduplicate else "link"

    if total_rows is None:
        total_rows = min_possible_edges
    elif total_rows == 0:
        raise ValueError("At least one edge must be generated")
    elif total_rows < min_possible_edges:
        raise ValueError(
            dedent(f"""
            Cannot generate {total_rows:,} {mode} edges with {num_components:,}
            components.
            Min edges is {min_possible_edges:,} for nodes given.
            Either decrease the number of nodes, increase the number of components, 
            or increase the total edges requested.
            """)
        )
    elif total_rows > max_possible_edges:
        raise ValueError(
            dedent(f"""
            Cannot generate {total_rows:,} {mode} edges with {num_components:,}
            components. 
            Max edges is {max_possible_edges:,} for nodes given.
            Either increase the number of nodes, decrease the number of components, 
            or decrease the total edges requested.
            """)
        )

    n_extra_edges = total_rows - min_possible_edges

    # Create seeded random number generator
    rng = np.random.default_rng(seed=seed)

    # Convert probability range to integers (60-80 for 0.60-0.80)
    prob_min = int(prob_range[0] * 100)
    prob_max = int(prob_range[1] * 100)

    # Split values into completely separate groups for each component
    left_components = np.array_split(np.array(left_values), num_components)
    right_components = np.array_split(np.array(right_values), num_components)
    # For each left-right component pair, the right equals the left rotated by one
    right_components = [np.roll(c, -1) for c in right_components]

    all_edges = []

    # Generate edges for each component
    for comp_idx in range(num_components):
        comp_left_values = left_components[comp_idx]
        comp_right_values = right_components[comp_idx]

        min_comp_nodes, max_comp_nodes = sorted(
            [len(comp_left_values), len(comp_right_values)]
        )

        # Ensure basic connectivity within the component by creating a spanning-tree
        base_edges = set()
        # For deduping (A B C) you just need (A - B) (B - C) (C - A)
        # which just needs matching pairwise the data and its rotated version.
        # For deduping, `min_comp_nodes` == `max_comp_nodes`
        if deduplicate:
            for i in range(min_comp_nodes - 1):
                small_n, large_n = sorted([comp_left_values[i], comp_right_values[i]])
                base_edges.add((small_n, large_n))
        else:
            # For linking (A B) and (C D E), we begin by adding (A - C) and (B - D)
            for i in range(min_comp_nodes):
                base_edges.add((comp_left_values[i], comp_right_values[i]))
            # we now add (C - B)
            for i in range(min_comp_nodes - 1):
                base_edges.add((comp_left_values[i + 1], comp_right_values[i]))
            # we now add (A - D)
            left_right_diff = max_comp_nodes - min_comp_nodes
            for i in range(left_right_diff):
                left_i, right_i = 0, min_comp_nodes + i
                if len(comp_right_values) < len(comp_left_values):
                    left_i, right_i = min_comp_nodes + i, 0

                base_edges.add((comp_left_values[left_i], comp_right_values[right_i]))

        component_edges = list(base_edges)

        if n_extra_edges > 0:
            # Generate remaining random edges strictly within this component
            # TODO: this can certainly be optimised
            if deduplicate:
                all_possible_edges = list(
                    {
                        tuple(sorted([x, y]))
                        for x in comp_left_values
                        for y in comp_right_values
                        if x != y and tuple(sorted([x, y])) not in base_edges
                    }
                )
            else:
                all_possible_edges = list(
                    {
                        (x, y)
                        for x in comp_left_values
                        for y in comp_right_values
                        if x != y and (x, y) not in base_edges
                    }
                )
            max_new_edges = len(all_possible_edges)
            if max_new_edges >= n_extra_edges:
                edges_required = n_extra_edges
                n_extra_edges = 0
            else:
                edges_required = max_new_edges
                n_extra_edges -= max_new_edges

            extra_edges_idx = rng.choice(
                len(all_possible_edges), size=edges_required, replace=False
            )
            extra_edges = [
                e for i, e in enumerate(all_possible_edges) if i in extra_edges_idx
            ]
            component_edges += extra_edges
        random_probs = rng.integers(prob_min, prob_max + 1, size=len(component_edges))

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
        [left_array, right_array, prob_array],
        names=["left_id", "right_id", "probability"],
    )


class ModelMetrics(BaseModel):
    """Metrics for a generated model."""

    n_true_entities: int


class ModelDummy(BaseModel):
    """Complete representation of a generated dummy Model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: ModelMetadata
    data: pa.Table
    metrics: ModelMetrics


def model_factory(
    name: str | None = None,
    description: str | None = None,
    model_type: Literal["deduper", "linker"] | None = None,
    n_true_entities: int = 10,
    prob_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 42,
) -> ModelDummy:
    """Generate a complete dummy model.

    Args:
        name: Name of the model
        description: Description of the model
        type: Type of the model, one of 'deduper' or 'linker'
        n_true_entities: Number of true entities to generate
        prob_range: Range of probabilities to generate
        seed: Random seed for reproducibility

    Returns:
        SourceModel: A dummy model with generated data
    """
    generator = Faker()
    generator.seed_instance(seed)

    model_type = ModelType(model_type.lower() if model_type else "deduper")

    model = ModelMetadata(
        name=name or generator.word(),
        description=description or generator.sentence(),
        type=model_type,
        left_resolution=generator.word(),
        right_resolution=generator.word() if model_type == ModelType.LINKER else None,
    )

    left_values = range(n_true_entities)
    right_values = None

    if model.type == ModelType.LINKER:
        right_values = range(n_true_entities, n_true_entities * 2)

    probabilities = generate_dummy_probabilities(
        left_values=left_values,
        right_values=right_values,
        prob_range=prob_range,
        num_components=n_true_entities,
        seed=seed,
    )

    return ModelDummy(
        model=model,
        data=probabilities,
        metrics=ModelMetrics(n_true_entities=n_true_entities),
    )
