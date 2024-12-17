from matchbox.common.hash import IntMap


def test_core_intmap():
    im = IntMap()
    im.index(1, 2)
    im.index(3, 4)
    im.index(-1, -2)

    keys, values = im.export()
    assert keys == [-1, -2, -3]
    assert values == [(1, 2), (3, 4), (-1, -2)]


def test_salted_intmap():
    im = IntMap(salt=10)
    a = im.index(1, 2)
    b = im.index(3, 4)
    c = im.index(a, b)

    keys, values = im.export()

    assert keys == [a, b, c]
    assert a < 0 and a != -1
    assert b < 0 and a != -2
    assert c < 0 and c != -3

    assert values == [(1, 2), (3, 4), (a, b)]
