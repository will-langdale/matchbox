from matchbox.common.results import DisjointSet


class TestDisjointSet:
    def test_disjoint_set_empty(self):
        dsj = DisjointSet()

        assert dsj.get_components() == []

    def test_disjoint_set_same(self):
        dsj = DisjointSet()
        dsj.union(1, 1)

        assert dsj.get_components() == [{1}]

    def test_disjoint_set_redundant(self):
        dsj = DisjointSet()
        dsj.union(1, 2)

        assert dsj.get_components() == [{1, 2}]

        dsj.union(2, 1)

        assert dsj.get_components() == [{1, 2}]

    def test_disjoint_set_union(self):
        dsj = DisjointSet()
        dsj.union(1, 2)
        dsj.union(3, 4)
        dsj.union(5, 6)

        assert sorted(dsj.get_components()) == [{1, 2}, {3, 4}, {5, 6}]

        dsj.union(2, 3)
        dsj.union(4, 5)

        assert dsj.get_components() == [{1, 2, 3, 4, 5, 6}]
