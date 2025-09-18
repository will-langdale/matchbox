"""Objects to define a DAG which indexes, deduplicates and links data."""

import datetime
from collections import defaultdict
from typing import Self

from matchbox.client.models import Model
from matchbox.client.queries import Query
from matchbox.client.sources import Source
from matchbox.common.logging import logger


class DAG:
    """Self-sufficient pipeline of indexing, deduping and linking steps."""

    def __init__(self, name: str, new: bool = False):
        """Initialises empty DAG."""
        self.name = name
        self.nodes: dict[str, Source | Model] = {}
        self.graph: dict[str, list[str]] = {}

    def _check_dag(self, dag: Self):
        if self != dag:
            raise ValueError("Cannot mix DAGs")

    def _add_step(self, step: Source | Model) -> None:
        """Validate and add sources and models to DAG."""
        self._check_dag(step.dag)

        if step.name in self.nodes:
            raise ValueError(f"Name '{step.name}' is already taken in the DAG")
        if isinstance(step, Model):
            self._check_dag(step.left_query.dag)
            if step.right_query:
                self._check_dag(step.right_query.dag)

            for resolution_name in step.dependencies:
                if resolution_name not in self.nodes:
                    raise ValueError(f"Step {resolution_name} not added to DAG")
            self.graph[step.name] = step.parents
        else:
            self.graph[step.name] = []

        self.nodes[step.name] = step

    def _build_inverse_graph(self) -> tuple[dict[str, list[str]], str]:
        """Build inverse graph and find the apex node.

        Returns:
            tuple: (inverse_graph, apex_node)

                * inverse_graph: Dictionary mapping nodes to their parent nodes
                * apex_node: The root node of the DAG

        Raises:
            ValueError: If the DAG has multiple disconnected components
        """
        inverse_graph = defaultdict(list)
        for node in self.graph:
            for neighbor in self.graph[node]:
                inverse_graph[neighbor].append(node)

        apex_nodes = {node for node in self.graph if node not in inverse_graph}
        if len(apex_nodes) > 1:
            raise ValueError("Some models or sources are disconnected")
        elif not apex_nodes:
            raise ValueError("No root node found, DAG might contain cycles")
        else:
            return inverse_graph, apex_nodes.pop()

    def source(self, *args, **kwargs) -> Source:
        """Create Source and add it to the DAG."""
        source = Source(*args, **kwargs, dag=self)
        self._add_step(source)

        return source

    def model(self, *args, **kwargs) -> Model:
        """Create Model and add it to the DAG."""
        model = Model(*args, **kwargs, dag=self)
        self._add_step(model)

        return model

    def query(self, *args, **kwargs) -> Query:
        """Create Query object."""
        return Query(*args, **kwargs, dag=self)

    def draw(
        self,
        start_time: datetime.datetime | None = None,
        doing: str | None = None,
        skipped: list[str] | None = None,
    ) -> str:
        """Create a string representation of the DAG as a tree structure.

        If `start_time` is provided, it will show the status of each node
        based on the last run time. The status indicators are:

        * ✅ Done
        * 🔄 Working
        * ⏸️ Awaiting
        * ⏭️ Skipped

        Args:
            start_time: Start time of the DAG run. Used to calculate node status.
            doing: Name of the node currently being processed (if any).
            skipped: List of node names that were skipped.

        Returns:
            String representation of the DAG with status indicators.
        """
        _, root_name = self._build_inverse_graph()
        skipped = skipped or []

        def _get_node_status(name: str) -> str:
            """Determine the status indicator for a node."""
            if name in skipped:
                return "⏭️"
            elif doing and name == doing:
                return "🔄"
            elif (
                (node := self.nodes.get(name))
                and node.last_run
                and node.last_run > start_time
            ):
                return "✅"
            else:
                return "⏸️"

        # Add status indicator if start_time is provided
        if start_time is not None:
            status = _get_node_status(root_name)
            result = [f"{status} {root_name}"]
        else:
            result = [root_name]

        visited = set([root_name])

        def format_children(node: str, prefix=""):
            """Recursively format the children of a node."""
            children = []
            # Get all outgoing edges from this node
            for target in self.graph.get(node, []):
                if target not in visited:
                    children.append(target)
                    visited.add(target)

            # Format each child
            for i, child in enumerate(children):
                is_last = i == len(children) - 1

                # Add status indicator if start_time is provided
                if start_time is not None:
                    status = _get_node_status(child)
                    child_display = f"{status} {child}"
                else:
                    child_display = child

                if is_last:
                    result.append(f"{prefix}└── {child_display}")
                    format_children(child, prefix + "    ")
                else:
                    result.append(f"{prefix}├── {child_display}")
                    format_children(child, prefix + "│   ")

        format_children(root_name)

        return "\n".join(result)

    def run_and_sync(
        self,
        full_rerun: bool = False,
        start: str | None = None,
        finish: str | None = None,
    ):
        """Run entire DAG and send results to server."""
        start_time = datetime.datetime.now()

        # Determine order of execution steps
        _, apex = self._build_inverse_graph()

        def depth_first(node: str, sequence: list):
            sequence.append(node)
            for neighbour in self.graph[node]:
                if neighbour not in sequence:
                    depth_first(neighbour, sequence)

        inverse_sequence = []
        depth_first(apex, inverse_sequence)
        sequence = list(reversed(inverse_sequence))

        # Identify skipped nodes
        skipped_nodes = []
        if start:
            try:
                start_index = sequence.index(start)
                skipped_nodes = sequence[:start_index]
            except ValueError as e:
                raise ValueError(f"Step {start} not in DAG") from e
        else:
            start_index = 0

        # Determine end index
        if finish:
            try:
                end_index = sequence.index(finish) + 1
                skipped_nodes.extend(sequence[end_index:])
            except ValueError as e:
                raise ValueError(f"Step {finish} not in DAG") from e
        else:
            end_index = len(sequence)

        sequence = sequence[start_index:end_index]
        if not full_rerun:
            # Exclude nodes that already run, unless a full re-run is forced
            already_run = [node for node in sequence if self.nodes[node].last_run]
            skipped_nodes.extend(already_run)
            sequence = [node for node in sequence if node not in already_run]

        for step_name in sequence:
            node = self.nodes[step_name]

            logger.info(
                "\n"
                + self.draw(
                    start_time=start_time, doing=node.name, skipped=skipped_nodes
                )
            )
            try:
                node.run()
                node.sync()
            except Exception as e:
                logger.error(f"❌ {node.name} failed: {e}")
                raise e

        logger.info("\n" + self.draw(start_time=start_time, skipped=skipped_nodes))
