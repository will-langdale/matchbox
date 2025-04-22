import pyarrow as pa

from matchbox.common.hash import IntMap, hash_arrow_table


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


def test_hash_arrow_table():
    a = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    # Column order should not matter
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
    # Column and row order should not matter
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
    # If column name change their order, the hash should change
    g = pa.Table.from_pydict(
        {
            "b": [1, 2, 3],
            "a": [4, 5, 6],
        }
    )
    # List columns are handled
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
    # Binary columns are handled, including non-UTF-8 bytes
    j = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [b"abc", None, bytes([255, 254, 253])],
        }
    )

    h_a = hash_arrow_table(a)
    h_a1 = hash_arrow_table(a)
    h_b = hash_arrow_table(b)
    h_c = hash_arrow_table(c)
    h_d = hash_arrow_table(d)
    h_e = hash_arrow_table(e)
    h_f = hash_arrow_table(f)
    h_g = hash_arrow_table(g)
    h_h = hash_arrow_table(h)
    h_i = hash_arrow_table(i)
    h_j = hash_arrow_table(j)

    # Basic type check
    assert isinstance(h_a, bytes)
    # Basic invariance checks
    assert h_a == h_a1 == h_b == h_c == h_d
    # Different data = different hashes
    assert h_a != h_e
    assert h_a != h_f
    assert h_a != h_g
    assert h_a != h_j
    # List type table should be consistent regardless of column order
    assert h_h == h_i
