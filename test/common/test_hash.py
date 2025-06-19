import polars as pl
import pyarrow as pa
import pytest

from matchbox.common.hash import (
    Cluster,
    HashMethod,
    IntMap,
    hash_arrow_table,
    hash_rows,
)
from matchbox.common.transform import DisjointSet


def test_intmap_basic():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2)
    b = im1.index(3, 4)
    c = im1.index(a, b)

    assert len({a, b, c}) == 3
    assert max(a, b, c) < 0


def test_intmap_same():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2)
    b = im1.index(3, 4)
    c = im1.index(a, b)

    im2 = IntMap(salt=10)
    x = im2.index(1, 2)
    y = im2.index(3, 4)
    z = im2.index(a, b)

    assert (a, b, c) == (x, y, z)


def test_intmap_different():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2)
    b = im1.index(3, 4)
    c = im1.index(a, b)

    im2 = IntMap(salt=11)
    x = im2.index(1, 2)
    y = im2.index(3, 4)
    z = im2.index(a, b)

    for v1, v2 in zip([a, b, c], [x, y, z], strict=True):
        assert v1 != v2


def test_intmap_unordered():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2, 3)
    b = im1.index(3, 1, 2)

    assert a == b


@pytest.mark.parametrize(
    ["method"],
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
def test_hash_rows(method: HashMethod):
    data = pl.DataFrame(
        {
            "string_col": ["abc", "def", "ghi"],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "struct_col": [{"a": 1, "b": "x"}, {"a": 2, "b": None}, {"a": 3, "b": "z"}],
            "binary_col": [b"data1", b"data2", b"data3"],
        }
    )

    assert isinstance(data["string_col"].dtype, pl.String)
    assert isinstance(data["int_col"].dtype, pl.Int64)
    assert isinstance(data["float_col"].dtype, pl.Float64)
    assert isinstance(data["struct_col"].dtype, pl.Struct)
    assert isinstance(data["binary_col"].dtype, pl.Binary)

    hash_rows(data, columns=data.columns, method=method)


@pytest.mark.parametrize(
    ["method"],
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
def test_hash_arrow_table(method: HashMethod):
    a = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    # Field order should not matter
    b = pa.Table.from_pydict(
        {
            "b": [4, 5, 6],
            "a": [1, 2, 3],
        }
    )
    # Row order should not matter
    c = pa.Table.from_pydict(
        {
            "a": [3, 2, 1],
            "b": [6, 5, 4],
        }
    )
    # Field and row order should not matter
    d = pa.Table.from_pydict(
        {
            "b": [6, 5, 4],
            "a": [3, 2, 1],
        }
    )
    # Empty table should have a different hash
    e = pa.Table.from_pydict(
        {
            "a": [],
            "b": [],
        }
    )
    # Different row data should have a different hash
    f = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 7],
        }
    )
    # If field name change their order, the hash should change
    g = pa.Table.from_pydict(
        {
            "b": [1, 2, 3],
            "a": [4, 5, 6],
        }
    )
    # List fields are handled
    h = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [[1, 2], [3, 4], [5, 6]],
        }
    )
    # List order doesn't matter
    i = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [[2, 1], [4, 3], [6, 5]],
        }
    )
    # Binary fields are handled, including non-UTF-8 bytes
    j = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [b"abc", None, bytes([255, 254, 253])],
        }
    )

    h_a = hash_arrow_table(a, method=method)
    h_a1 = hash_arrow_table(a, method=method)
    h_b = hash_arrow_table(b, method=method)
    h_c = hash_arrow_table(c, method=method)
    h_d = hash_arrow_table(d, method=method)
    h_e = hash_arrow_table(e, method=method)
    h_f = hash_arrow_table(f, method=method)
    h_g = hash_arrow_table(g, method=method)
    h_h = hash_arrow_table(h, method=method)
    h_i = hash_arrow_table(i, method=method)
    h_j = hash_arrow_table(j, method=method)

    # Basic type check
    assert isinstance(h_a, bytes)
    # Basic invariance checks
    assert h_a == h_a1 == h_b == h_c == h_d
    # Different data = different hashes
    assert h_a != h_e
    assert h_a != h_f
    assert h_a != h_g
    assert h_a != h_j
    # List type table should be consistent regardless of field order
    assert h_h == h_i


@pytest.mark.parametrize(
    ["method"],
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
def test_struct_json_hashing(method: HashMethod):
    """Test that struct/JSON data can be properly hashed."""

    # Basic struct test
    a = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 35},
            ],
        }
    )

    assert isinstance(pl.from_arrow(a)["metadata"].dtype, pl.Struct)

    # Same data but different struct serialization
    b = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"age": 30, "name": "Alice"},
                {"age": 25, "name": "Bob"},
                {"age": 35, "name": "Charlie"},
            ],
        }
    )

    # Different data in structs
    c = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"name": "Alice", "age": 31},  # Changed age
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 35},
            ],
        }
    )

    # Nested structs
    d = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"name": "Alice", "details": {"city": "New York", "active": True}},
                {"name": "Bob", "details": {"city": "Boston", "active": False}},
                {"name": "Charlie", "details": {"city": "Chicago", "active": True}},
            ],
        }
    )

    # Test basic struct hashing
    h_a = hash_arrow_table(a, method=method)
    h_a1 = hash_arrow_table(a, method=method)
    h_b = hash_arrow_table(b, method=method)
    h_c = hash_arrow_table(c, method=method)
    h_d = hash_arrow_table(d, method=method)

    # Basic type check
    assert isinstance(h_a, bytes)

    # Basic equality check
    assert h_a == h_a1 == h_b
    # Difference checks
    assert h_a != h_c
    assert h_a != h_d


def collect_all_leaves(cluster: Cluster) -> set[Cluster]:
    """Recursively collect all leaf nodes from a cluster."""
    if cluster.leaves is None:
        return {cluster}

    leaves = set()
    for leaf in cluster.leaves:
        leaves.update(collect_all_leaves(leaf))
    return leaves


class TestClusterHierarchy:
    """Tests for using Cluster objects with DisjointSet to build hierarchies."""

    @pytest.fixture
    def intmap(self) -> IntMap:
        """Create a fresh IntMap for each test."""
        return IntMap(salt=10)

    @pytest.fixture
    def leaf_nodes(self, intmap: IntMap) -> list[Cluster]:
        """Create six leaf nodes for testing."""
        return [
            Cluster(
                intmap=intmap,
                probability=None,
                leaves=None,
                id=i,
                hash=f"hash{i}".encode(),
            )
            for i in range(1, 7)
        ]

    def test_add_leaves_to_disjoint_set(self, leaf_nodes: list[Cluster]):
        """Test adding leaf nodes to a DisjointSet."""
        # Create a DisjointSet and add leaves
        djs = DisjointSet[Cluster]()
        for node in leaf_nodes:
            djs.add(node)

        # Verify all nodes are in separate components
        components = djs.get_components()
        assert len(components) == 6
        for component in components:
            assert len(component) == 1

    def test_union_leaves_in_disjoint_set(self, leaf_nodes: list[Cluster]):
        """Test creating unions of leaf nodes in DisjointSet."""
        djs = DisjointSet[Cluster]()
        for node in leaf_nodes:
            djs.add(node)

        # Create three pairs
        djs.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        djs.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        djs.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        # Verify we now have three components
        components = djs.get_components()
        assert len(components) == 3

        # Verify each component has the expected size
        component_sizes = [len(comp) for comp in components]
        assert sorted(component_sizes) == [2, 2, 2]

    def test_create_clusters_from_components(self, leaf_nodes: list[Cluster]):
        """Test creating new clusters from DisjointSet components."""
        # Add leaves to DisjointSet and create unions
        djs = DisjointSet[Cluster]()
        for node in leaf_nodes:
            djs.add(node)

        djs.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        djs.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        djs.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        # Create clusters from components
        components = djs.get_components()
        clusters: list[Cluster] = []
        for component in components:
            component_list = list(component)
            if len(component_list) == 1:
                cluster = component_list[0]
            else:
                cluster = Cluster.combine(component_list, probability=100)
            clusters.append(cluster)

        # Verify we have three clusters
        assert len(clusters) == 3

        # Verify each non-leaf cluster has exactly two leaves
        for cluster in clusters:
            if cluster.leaves is not None:
                assert len(cluster.leaves) == 2

        # Verify probability is as expected for non-leaf clusters
        for cluster in clusters:
            if cluster.leaves is not None:
                assert cluster.probability == 100

    def test_level1_clusters_in_disjoint_set(self, leaf_nodes: list[Cluster]):
        """Test using level-1 clusters in another DisjointSet."""
        # Create level-1 clusters as in previous test
        djs1 = DisjointSet[Cluster]()
        for node in leaf_nodes:
            djs1.add(node)

        djs1.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        djs1.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        djs1.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        components = djs1.get_components()
        level1_clusters = []
        for component in components:
            component_list = list(component)
            if len(component_list) == 1:
                cluster = component_list[0]
            else:
                cluster = Cluster.combine(component_list)
            level1_clusters.append(cluster)

        # Add these clusters to a new DisjointSet
        djs2 = DisjointSet[Cluster]()
        for cluster in level1_clusters:
            djs2.add(cluster)

        # Verify initial state
        assert len(djs2.get_components()) == 3

        # Create a union of first two clusters
        djs2.union(level1_clusters[0], level1_clusters[1])

        # Verify we now have two components
        components2 = djs2.get_components()
        assert len(components2) == 2

        # One component should have 2 clusters, the other should have 1
        component_sizes = [len(comp) for comp in components2]
        assert sorted(component_sizes) == [1, 2]

    def test_leaf_preservation_in_hierarchy(self, leaf_nodes: list[Cluster]):
        """Test that leaf nodes are preserved in the cluster hierarchy."""
        # Follow the same steps as in previous tests to build a hierarchy
        # Step 1: Create level-1 clusters
        djs1 = DisjointSet[Cluster]()
        for node in leaf_nodes:
            djs1.add(node)

        djs1.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        djs1.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        djs1.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        components1 = djs1.get_components()
        level1_clusters = []
        for component in components1:
            component_list = list(component)
            if len(component_list) == 1:
                cluster = component_list[0]
            else:
                cluster = Cluster.combine(component_list, probability=90)
            level1_clusters.append(cluster)

        # Verify probability as expected
        for cluster in level1_clusters:
            assert cluster.probability == 90

        # Step 2: Union level-1 clusters
        djs2 = DisjointSet[Cluster]()
        for cluster in level1_clusters:
            djs2.add(cluster)

        djs2.union(level1_clusters[0], level1_clusters[1])

        # Find the component with multiple clusters
        larger_component = None
        for component in djs2.get_components():
            if len(component) > 1:
                larger_component = component
                break

        assert larger_component is not None

        # Create a level-2 cluster from the larger component
        level2_cluster = Cluster.combine(larger_component, probability=80)

        # Verify the level-2 cluster contains exactly the leaf nodes 1, 2, 3, and 4
        expected_leaves = set(leaf_nodes[:4])  # Nodes 1-4

        # Collect all leaf nodes from the level-2 cluster
        actual_leaves = collect_all_leaves(level2_cluster)

        # Verify leaf nodes match
        assert actual_leaves == expected_leaves
        assert len(actual_leaves) == 4
        # Verify probability as expected
        assert level2_cluster.probability == 80

    def test_combine_with_single_cluster(self, leaf_nodes: list[Cluster]):
        """Test that combine works correctly with a single cluster."""
        # Create a single-element list
        clusters = [leaf_nodes[0]]

        # Combine
        result = Cluster.combine(clusters, probability=None)

        # Should return the original cluster
        assert result is leaf_nodes[0]
        # Verify probability as expected
        assert result.probability is None

    def test_combine_with_leaf_and_non_leaf(
        self, intmap: IntMap, leaf_nodes: list[Cluster]
    ):
        """Test combine with a mix of leaf and non-leaf clusters."""
        # Create a non-leaf cluster
        non_leaf = Cluster(intmap=intmap, leaves=[leaf_nodes[0], leaf_nodes[1]])

        # Combine with a leaf node
        result = Cluster.combine([non_leaf, leaf_nodes[2]], probability=70)

        # Should contain three leaves
        assert len(collect_all_leaves(result)) == 3

        # The specific leaves should be 1, 2, and 3
        expected = {leaf_nodes[0], leaf_nodes[1], leaf_nodes[2]}
        assert collect_all_leaves(result) == expected

        # Verify probability as expected
        assert result.probability == 70

    def test_hash_consistency_regardless_of_order(self, leaf_nodes: list[Cluster]):
        """Test that the hash is consistent regardless of the order of leaves."""
        # Create two clusters with the same leaves but in different order
        cluster1 = Cluster.combine([leaf_nodes[0], leaf_nodes[1]])
        cluster2 = Cluster.combine([leaf_nodes[1], leaf_nodes[0]])

        # Verify that both clusters have the same hash
        assert cluster1.hash == cluster2.hash

        # Test with more complex combinations
        cluster3 = Cluster.combine([leaf_nodes[0], leaf_nodes[1], leaf_nodes[2]])
        cluster4 = Cluster.combine([leaf_nodes[2], leaf_nodes[0], leaf_nodes[1]])
        cluster5 = Cluster.combine([leaf_nodes[1], leaf_nodes[2], leaf_nodes[0]])

        # All permutations should have the same hash
        assert cluster3.hash == cluster4.hash
        assert cluster4.hash == cluster5.hash

    def test_hash_based_on_leaf_hashes_only(self, leaf_nodes: list[Cluster]):
        """Test that cluster hashes are based on leaf node hashes only."""
        # Create two different paths to the same set of leaf nodes

        # Path 1: Direct combination of three leaves
        direct = Cluster.combine([leaf_nodes[0], leaf_nodes[1], leaf_nodes[2]])

        # Path 2: Hierarchical combination
        intermediate = Cluster.combine([leaf_nodes[0], leaf_nodes[1]])
        hierarchical = Cluster.combine([intermediate, leaf_nodes[2]])

        # Both should have the same hash since they contain the same leaf nodes
        assert direct.hash == hierarchical.hash

        # Create a completely different structure with the same leaves
        # This time using combine with all leaves
        combined = Cluster.combine([leaf_nodes[0], leaf_nodes[1], leaf_nodes[2]])

        # Should still have the same hash
        assert direct.hash == combined.hash

    def test_hash_with_provided_hash_value(
        self, intmap: IntMap, leaf_nodes: list[Cluster]
    ):
        """Test that providing a hash value overrides the calculated hash."""
        # Create a cluster with a provided hash
        custom_hash = b"custom_hash_value"
        cluster = Cluster(
            intmap=intmap,
            leaves=[leaf_nodes[0], leaf_nodes[1]],
            hash=custom_hash,
        )

        # Verify the hash is the custom one
        assert cluster.hash == custom_hash

        # Verify it's different from what would have been calculated
        auto_cluster = Cluster.combine([leaf_nodes[0], leaf_nodes[1]])
        assert cluster.hash != auto_cluster.hash

    def test_hash_generation_for_complex_hierarchy(self, leaf_nodes: list[Cluster]):
        """Test that hash generation works correctly for complex hierarchies."""
        # Create level 1 clusters
        cluster_a = Cluster.combine([leaf_nodes[0], leaf_nodes[1]])
        cluster_b = Cluster.combine([leaf_nodes[2], leaf_nodes[3]])
        cluster_c = Cluster.combine([leaf_nodes[4], leaf_nodes[5]])

        # Create level 2 clusters
        level2_a = Cluster.combine([cluster_a, cluster_b])
        level2_b = cluster_c

        # Create top level cluster
        top_level = Cluster.combine([level2_a, level2_b])

        # Verify the top level hash is derived from all leaf nodes
        # by comparing with a direct combination
        direct_combination = Cluster.combine(leaf_nodes)

        # Both should have the same hash since they contain the same leaf nodes
        assert top_level.hash == direct_combination.hash

        # Check that the hash is consistent when using different intermediate groupings
        alternate_grouping = Cluster.combine(
            [
                Cluster.combine([leaf_nodes[0], leaf_nodes[1], leaf_nodes[2]]),
                Cluster.combine([leaf_nodes[3], leaf_nodes[4], leaf_nodes[5]]),
            ]
        )
        assert top_level.hash == alternate_grouping.hash
