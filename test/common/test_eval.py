import polars as pl

from matchbox.common.eval import (
    eval_data_to_pairs,
    filter_eval_pairs,
    model_root_leaf_to_pairs,
)


def test_filter_eval_pairs():
    # Only leaves shared by all are 1,2,3,4
    validation_pairs = {(1, 3), (2, 4), (5, 6)}
    model1_pairs = {(1, 2), (3, 4), (5, 6)}
    model2_pairs = {(1, 10), (2, 11), (3, 12), (4, 14)}

    filtered_validation, filtered_model1, filtered_model2 = filter_eval_pairs(
        validation_pairs, model1_pairs, model2_pairs
    )
    assert filtered_validation == {(1, 3), (2, 4)}
    assert filtered_model1 == {(1, 2), (3, 4)}
    assert filtered_model2 == set()


def test_model_root_leaf_to_pairs():
    root_leaf = pl.DataFrame(
        [
            # TODO: check this is the right representation for singleton
            {"root": 0, "leaf": 0},
            # Descending order to check sorting
            {"root": 10, "leaf": 3},
            {"root": 10, "leaf": 2},
            {"root": 10, "leaf": 1},
        ]
    )

    pairs = model_root_leaf_to_pairs(root_leaf)

    assert pairs == {(1, 2), (1, 3), (2, 3)}


def test_eval_data_to_pairs():
    # In this test:
    # cluster IDs < 100 are source clusters
    # 1xx clusters IDs are model clusters
    # 2xx clusters IDs are new judgement clusters (splintered model clusters)

    judgements = pl.DataFrame(
        [
            # A) cluster confirmed as is
            {"shown": 100, "endorsed": 100},  # (12)->(12)
            # B) cluster splintered
            {"shown": 101, "endorsed": 200},  # (345)->(34)
            {"shown": 101, "endorsed": 201},  # (345)->(5)
            # C) mixed and repeated opinions
            {"shown": 102, "endorsed": 102},  # (67)->(67)
            {"shown": 102, "endorsed": 102},  # repeated...
            {"shown": 102, "endorsed": 202},  # ...and negated: (67)->(6)
            {"shown": 102, "endorsed": 203},  # (67)->(7)
            # D) judgement has seen less than a previous one
            {"shown": 103, "endorsed": 103},  # (89) -> (89)
            {"shown": 104, "endorsed": 104},  # (8,9,10) -> (8,9,10)
        ]
    )
    # TODO: what about singletons?
    # TODO: what about multiple keys per cluster?
    expansion = pl.DataFrame(
        [
            # Shown clusters
            {"root": 100, "leaves": [1, 2]},
            {"root": 101, "leaves": [3, 4, 5]},
            {"root": 102, "leaves": [6, 7]},
            {"root": 103, "leaves": [8, 9]},
            {"root": 104, "leaves": [8, 9, 10]},
            # New "splintered" clusters
            {"root": 200, "leaves": [3, 4]},
            {"root": 201, "leaves": [5]},
            {"root": 202, "leaves": [6]},
            {"root": 203, "leaves": [7]},
        ]
    )

    pairs, weights = eval_data_to_pairs(judgements, expansion)
    assert pairs == {(1, 2), (3, 4), (3, 5), (4, 5), (6, 7), (8, 9), (8, 10), (9, 10)}

    for positive_pair in [(1, 2), (3, 4), (8, 9), (8, 10), (9, 10)]:
        assert weights[positive_pair] == 1

    for negative_pair in [(3, 5), (4, 5)]:
        assert weights[negative_pair] == 0

    # And one of them has mixed opinions
    assert weights[(6, 7)] == 2 / 3
