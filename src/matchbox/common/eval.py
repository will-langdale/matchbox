"""Common operations to produce model evaluation scores."""

from collections import Counter
from itertools import chain, combinations
from typing import TypeAlias

import polars as pl
from pyarrow import Table
from pydantic import BaseModel, Field

from matchbox.common.dtos import ModelResolutionName

Pair: TypeAlias = tuple[int, int]
Pairs: TypeAlias = set[Pair]
PairWeights: TypeAlias = dict[Pair, float]

PrecisionRecall: TypeAlias = tuple[float, float]
ModelComparison: TypeAlias = dict[ModelResolutionName, PrecisionRecall]


class Judgement(BaseModel):
    """User determination on how to group source clusters from a model cluster."""

    user_id: int
    shown: int = Field(description="ID of the model cluster shown to the user")
    endorsed: list[list[int]] = Field(
        description="""Groups of source cluster IDs that user thinks belong together"""
    )


def filter_eval_pairs(*pairs: Pairs) -> list[Pairs]:
    """Filter sequence of pair sets so that each set contains leaves shared by all."""
    leaves_per_set = [set(list(chain.from_iterable(p))) for p in pairs]
    shared_leaves = set.intersection(*leaves_per_set)

    filtered_sets = []
    for p in pairs:
        filtered_sets.append(
            {(a, b) for a, b in p if a in shared_leaves and b in shared_leaves}
        )
    return filtered_sets


def precision_recall(
    models_root_leaf: list[Table], judgements: Table, expansion: Table
) -> list[PrecisionRecall]:
    """Compute precision and recall scores from models and eval data."""
    all_model_pairs: list[Pairs] = [
        model_root_leaf_to_pairs(pl.from_arrow(mrl)) for mrl in models_root_leaf
    ]

    validation_pairs, validation_weights = eval_data_to_pairs(
        pl.from_arrow(judgements), pl.from_arrow(expansion)
    )

    filtered_pairs = filter_eval_pairs(*all_model_pairs, validation_pairs)
    all_model_pairs = filtered_pairs[:-1]
    validation_pairs = filtered_pairs[-1]
    # TODO: complete this function


def model_root_leaf_to_pairs(root_leaf: pl.DataFrame) -> Pairs:
    """Convert root-leaf representation to sorted pairs.

    For example, a model cluster (123) will give pairs (12),(13),(23).
    """
    clusters = (
        root_leaf.group_by("root")
        .agg(pl.col("leaf").alias("leaves"))
        .select("leaves")
        .to_series()
        .to_list()
    )
    pairs = set()
    for c in clusters:
        pairs.update(combinations(sorted(c), r=2))

    return pairs


def eval_data_to_pairs(
    judgements: pl.DataFrame, expansion: pl.DataFrame
) -> tuple[PairWeights, Pairs]:
    """Convert user judgements to pairs and weights.

    In general, pairs include all (sorted) pair-wise combinations of elements in a list
    of cluster IDs. For example, (123) will give us (1,2), (1,3), (2,3). We, however,
    need to capture the difference between when a user is shown (12) and endorses (12),
    vs. when the user is shown (123) and endorses (12). In the second case, the user
    implies a negative judgement over pairs (1,3) and (2,3). We use weights to map all
    potential pairs (corresponding to the cluster a user was shown) to the proportion of
    times those pairs were confirmed (endorsed) by a user.

    This function relies on the input data being well-formed. For example,

    - All shown and endorsed cluster IDs need to have an expansion.
    - No partial splintered clusters, i.e. if (123)->(12) is in the
    data, (123)->(3) must be as well.

    Args:
        judgements: Dataframe following `matchbox.common.arrow.SCHEMA_JUDGEMENTS`.
        expansion: Dataframe following `matchbox.common.arrow.SCHEMA_CLUSTER_EXPANSION`.

    Returns:
        Tuple of:

        - Weights representing percentage of positive pairs.
        - Set of pairs, both positive and negative.
    """
    expanded_judgements = (
        judgements.join(expansion, left_on="shown", right_on="root")
        .rename({"leaves": "shown_leaves"})
        .join(expansion, left_on="endorsed", right_on="root")
        .rename({"leaves": "endorsed_leaves"})
    )
    assert len(expanded_judgements) == len(judgements)

    # Track endorsed pairs
    positive_count = Counter()
    # We track potential (shown) pairs differently. We could be double-counting,
    # since if a shown cluster is splintered, the same shown element will appear
    # multiple times for distinct endorsed clusters, e.g. (123)->(12);(123)->(3).
    # Instead, we only add to the potential count the ratio between the lengths of
    # endorsed and of shown. E.g. (123)->(12) means +2/3; (123)->(3) means +1/3
    potential_count: dict[Pair, float] = {}

    for shown, endorsed in zip(
        expanded_judgements["shown_leaves"].to_list(),
        expanded_judgements["endorsed_leaves"].to_list(),
        strict=True,
    ):
        positive_count.update(combinations(sorted(endorsed), r=2))

        potential_pairs = list(combinations(sorted(shown), r=2))
        potential_count.update(
            {
                p: potential_count.get(p, 0) + (len(endorsed) / len(shown))
                for p in potential_pairs
            }
        )

    all_pairs = set(potential_count.keys())
    pair_weights = {
        pair: positive_count[pair] / potential_count[pair] for pair in all_pairs
    }

    return all_pairs, pair_weights
