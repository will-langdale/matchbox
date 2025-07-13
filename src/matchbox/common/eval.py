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


def precision_recall(
    models_root_leaf: list[Table], judgements: Table, expansion: Table
) -> list[PrecisionRecall]:
    """From models and eval data, compute scores inspired by precision-recall.

    This function does the following:

    - Call `model_root_leaf_to_pairs()` and `eval_data_to_pairs()` to convert clusters
        to implied pair-wise connections for the models and judgements.
        This includes the pairs that a user was shown, but did not endorse.
        For judgements, this also finds the proportion of times a pair shown to users
        was actually endorsed.
    - Call `filter_eval_pairs` to find the pairs present in all models and in the
        judgements, so the comparison is fair.
    - Call `pairs_to_scores()` to computes scores similar to precision recall, but
        considering proportion of positive endorsement for all judgements.

    For more information on the implementation of each step, you can review the
    docstrings of the relevant functions.

    Args:
        models_root_leaf: list of tables with root and leaf columns, one per model.
            They must include all the clusters that resolve from a model, all the way
            to the original source clusters if no model in the lineage merged them.
        judgements: Dataframe following `matchbox.common.arrow.SCHEMA_JUDGEMENTS`.
        expansion: Dataframe following `matchbox.common.arrow.SCHEMA_CLUSTER_EXPANSION`.

    Returns:
        List of precision-recall inspired scores, one per model.
    """
    pairs_per_model: list[Pairs] = [
        model_root_leaf_to_pairs(pl.from_arrow(mrl)) for mrl in models_root_leaf
    ]

    validation_pairs, validation_weights = eval_data_to_pairs(
        pl.from_arrow(judgements), pl.from_arrow(expansion)
    )

    filtered_pairs = filter_eval_pairs(*pairs_per_model, validation_pairs)
    pairs_per_model = filtered_pairs[:-1]
    validation_pairs = filtered_pairs[-1]

    pr_scores: list[PrecisionRecall] = []

    for model_pairs in pairs_per_model:
        pr_scores.append(
            pairs_to_scores(model_pairs, validation_pairs, validation_weights)
        )

    return pr_scores


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


def pairs_to_scores(
    model_pairs: Pairs, validation_pairs: Pairs, validation_weights: PairWeights
) -> PrecisionRecall:
    """From model and validation pairs, compute scores inspired by precision-recall.

    Traditional precision-recall would use these formulae:

    P = relevant_model_pairs / all_model_pairs
    R = relevant_model_pairs / relevant_pairs

    We do that, except that instead of counting 1 for each validation pair,
    we weigh it by the proportion of positive endorsements. For example, let' say
    the model has (12), (13), (23), and the user judgements have (12), (13), ~(13),
    (23), ~(23). This means that user judgements are (12), (13), (23), but (13) and (23)
    both have a weight of 0.5. Hence:

    relevant_model_pairs = 1 + 0.5 + 0.5 = 2

    For the denumerator of P, we don't use weights, as we assume the model, once
    a threshold is set, makes certain judgements (we can of course variate the)
    threshold to compute a curve. Hence:

    all_model_pairs = 1 + 1 + 1 = 3

    Thus P equals 2/3, even though there are some judgements that would validate
    every one of the pairs proposed by the model, because the judgements have
    some uncertainty around them.

    For the denumerator of P, we do use weights, hence:

    relevant_pairs = 1 + 0.5 + 0.5 = 2

    So R equals 1. If we didn't use weights for the denumerator, the recall would equal
    2/3, which is not right: the model is doing as much retrieving as is possible to do!
    """
    true_positive_pairs = model_pairs & validation_pairs
    true_positive_weight = sum([validation_weights[p] for p in true_positive_pairs])
    precision_denumerator = len(model_pairs)
    recall_denumerator = sum([validation_weights[p] for p in validation_pairs])
    return (
        true_positive_weight / precision_denumerator,
        true_positive_weight / recall_denumerator,
    )


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
) -> tuple[Pairs, PairWeights]:
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

        - Set of pairs, both positive and negative.
        - Weights representing percentage of positive pairs.
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
