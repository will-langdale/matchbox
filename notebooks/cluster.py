from collections import defaultdict
from functools import lru_cache
from typing import Generic, Hashable, Iterator, TypeVar

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

T = TypeVar("T", bound=Hashable)


@lru_cache(maxsize=None)
def combine_integers(*n: int) -> int:
    """
    Combine n integers into a single negative integer.

    Used to create a symmetric deterministic hash of two integers that populates the
    range of integers efficiently and reduces the likelihood of collisions.

    Aims to vectorise amazingly when used in Arrow.

    Does this by:

    * Using a Mersenne prime as a modulus
    * Making negative integers positive with modulo, sped up with bitwise operations
    * Combining using symmetric operations with coprime multipliers

    Args:
        *args: Variable number of integers to combine

    Returns:
        A negative integer
    """
    P = 2147483647

    total = 0
    product = 1

    for x in sorted(n):
        x_pos = (x ^ (x >> 31)) - (x >> 31)
        total = (total + x_pos) % P
        product = (product * x_pos) % P

    result = (31 * total + 37 * product) % P

    return -result


class UnionFindWithDiff(Generic[T]):
    def __init__(self):
        self.parent: dict[T, T] = {}
        self.rank: dict[T, int] = {}
        self._shadow_parent: dict[T, T] = {}
        self._shadow_rank: dict[T, int] = {}
        self._pending_pairs: list[tuple[T, T]] = []

    def make_set(self, x: T) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: T, parent_dict: dict[T, T] | None = None) -> T:
        if parent_dict is None:
            parent_dict = self.parent

        if x not in parent_dict:
            self.make_set(x)
            if parent_dict is self._shadow_parent:
                self._shadow_parent[x] = x
                self._shadow_rank[x] = 0

        while parent_dict[x] != x:
            parent_dict[x] = parent_dict[parent_dict[x]]
            x = parent_dict[x]
        return x

    def union(self, x: T, y: T) -> None:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            self._pending_pairs.append((x, y))

            if self.rank[root_x] < self.rank[root_y]:
                root_x, root_y = root_y, root_x
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1

    def get_component(self, x: T, parent_dict: dict[T, T] | None = None) -> set[T]:
        if parent_dict is None:
            parent_dict = self.parent

        root = self.find(x, parent_dict)
        return {y for y in parent_dict if self.find(y, parent_dict) == root}

    def get_components(self, parent_dict: dict[T, T] | None = None) -> list[set[T]]:
        if parent_dict is None:
            parent_dict = self.parent

        components = defaultdict(set)
        for x in parent_dict:
            root = self.find(x, parent_dict)
            components[root].add(x)
        return list(components.values())

    def diff(self) -> Iterator[tuple[set[T], set[T]]]:
        """
        Returns differences including all pairwise merges that occurred since last diff,
        excluding cases where old_comp == new_comp.
        """
        # Get current state before processing pairs
        current_components = self.get_components()
        reported_pairs = set()

        # Process pending pairs
        for x, y in self._pending_pairs:
            # Find the final component containing the pair
            final_component = next(
                comp for comp in current_components if x in comp and y in comp
            )

            # Only report if the pair forms a proper subset of the final component
            pair_component = {x, y}
            if (
                pair_component != final_component
                and frozenset((frozenset(pair_component), frozenset(final_component)))
                not in reported_pairs
            ):
                reported_pairs.add(
                    frozenset((frozenset(pair_component), frozenset(final_component)))
                )
                yield (pair_component, final_component)

        self._pending_pairs.clear()

        # Handle initial state
        if not self._shadow_parent:
            self._shadow_parent = self.parent.copy()
            self._shadow_rank = self.rank.copy()
            return

        # Get old components
        old_components = self.get_components(self._shadow_parent)

        # Report changes between old and new states
        for old_comp in old_components:
            if len(old_comp) > 1:  # Only consider non-singleton old components
                sample_elem = next(iter(old_comp))
                new_comp = next(
                    comp for comp in current_components if sample_elem in comp
                )

                # Only yield if the components are different and this pair
                # hasn't been reported
                if (
                    old_comp != new_comp
                    and frozenset((frozenset(old_comp), frozenset(new_comp)))
                    not in reported_pairs
                ):
                    reported_pairs.add(
                        frozenset((frozenset(old_comp), frozenset(new_comp)))
                    )
                    yield (old_comp, new_comp)

        # Update shadow copy
        self._shadow_parent = self.parent.copy()
        self._shadow_rank = self.rank.copy()


def component_to_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pairwise probabilities into a hierarchical representation.
    Assumes data is pre-sorted by probability descending.

    Args:
        key: Group key (ignored in this implementation)
        df: DataFrame with columns ['left', 'right', 'probability']

    Returns:
        DataFrame with columns ['parent', 'child', 'probability']
    """
    hierarchy: list[tuple[int, int, float]] = []
    uf = UnionFindWithDiff[int]()

    for threshold in df["probability"].unique():
        current_probs = df[df["probability"] == threshold]

        for _, row in current_probs.iterrows():
            uf.union(row["left"], row["right"])
            parent = combine_integers(row["left"], row["right"])
            hierarchy.extend(
                [(parent, row["left"], threshold), (parent, row["right"], threshold)]
            )

        for old_comp, new_comp in uf.diff():
            if len(old_comp) > 1:
                parent = combine_integers(*new_comp)
                child = combine_integers(*old_comp)
                hierarchy.extend([(parent, child, threshold)])
            else:
                parent = combine_integers(*new_comp)
                hierarchy.extend([(parent, old_comp.pop(), threshold)])

    return pd.DataFrame(hierarchy, columns=["parent", "child", "probability"])


def component_to_hierarchy_pa(table: pa.Table) -> pa.Table:
    """
    Convert pairwise probabilities into a hierarchical representation.
    Assumes data is pre-sorted by probability descending.

    Args:
        key: Group key (ignored in this implementation)
        table: Arrow Table with columns ['left', 'right', 'probability']

    Returns:
        Arrow Table with columns ['parent', 'child', 'probability'] representing hierarchical merges
    """
    hierarchy: list[tuple[int, int, float]] = []
    uf = UnionFindWithDiff[int]()

    # Get unique probabilities
    probs = pc.unique(table["probability"])

    for threshold in probs:
        # Get current probability rows
        mask = pc.equal(table["probability"], threshold)
        current_probs = table.filter(mask)
        threshold_float = float(threshold.as_py())

        # Process each row
        for row in zip(
            current_probs["left"].to_numpy(),
            current_probs["right"].to_numpy(),
            strict=False,
        ):
            left, right = row
            uf.union(left, right)
            parent = combine_integers(left, right)
            hierarchy.extend(
                [(parent, left, threshold_float), (parent, right, threshold_float)]
            )

        # Process UnionFind diffs - exact same logic as original
        for old_comp, new_comp in uf.diff():
            if len(old_comp) > 1:
                parent = combine_integers(*new_comp)
                child = combine_integers(*old_comp)
                hierarchy.extend([(parent, child, threshold_float)])
            else:
                parent = combine_integers(*new_comp)
                hierarchy.extend([(parent, old_comp.pop(), threshold_float)])

    parents, children, probs = zip(*hierarchy, strict=False)
    return pa.table(
        {
            "parent": pa.array(parents, type=pa.int64()),
            "child": pa.array(children, type=pa.int64()),
            "probability": pa.array(probs, type=pa.float64()),
        }
    )


def process_component(comp: pa.Scalar, table: pa.Table) -> pa.Table:
    mask = pc.equal(table["component"], comp)
    component_table = table.filter(mask)
    return component_to_hierarchy_pa(component_table)
