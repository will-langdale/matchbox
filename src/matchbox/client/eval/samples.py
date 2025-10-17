"""Client-side helpers for retrieving and preparing evaluation samples."""

from typing import cast

import polars as pl
from pydantic import BaseModel, computed_field
from sqlalchemy.exc import OperationalError

from matchbox.client import _handler
from matchbox.client.dags import DAG
from matchbox.client.results import Results
from matchbox.common.dtos import ModelResolutionPath, SourceConfig
from matchbox.common.eval import Judgement, ModelComparison, precision_recall
from matchbox.common.exceptions import MatchboxSourceTableError


class EvaluationItem(BaseModel):
    """A cluster awaiting evaluation, with deduplicated column data."""

    model_config = {"arbitrary_types_allowed": True}

    cluster_id: int
    dataframe: pl.DataFrame  # Original raw data
    display_dataframe: pl.DataFrame  # Enhanced DataFrame for flexible rendering
    duplicate_groups: list[list[int]]  # Groups of leaf_ids with identical data
    display_columns: list[int]  # Representative leaf_id for each displayed column
    leaf_to_display_mapping: dict[int, int]  # actual leaf_id -> display_column_index
    assignments: dict[int, str] = {}  # display_column_index -> group_letter

    @computed_field
    @property
    def total_columns(self) -> int:
        """Total number of display columns in this cluster."""
        return len(self.display_columns)

    @computed_field
    @property
    def is_painted(self) -> bool:
        """True only if ALL columns have been assigned to groups."""
        return len(self.assignments) == self.total_columns

    def to_judgement(self, user_id: int) -> Judgement:
        """Convert assignments to Judgement format, expanding duplicate groups."""
        groups: dict[str, list[int]] = {}

        for display_col_index, group in self.assignments.items():
            bucket = groups.setdefault(group, [])
            if display_col_index < len(self.duplicate_groups):
                bucket.extend(self.duplicate_groups[display_col_index])

        assigned_display_cols = set(self.assignments.keys())
        unassigned_leaf_ids: list[int] = []
        for display_col_index in range(len(self.duplicate_groups)):
            if display_col_index not in assigned_display_cols:
                unassigned_leaf_ids.extend(self.duplicate_groups[display_col_index])

        if unassigned_leaf_ids:
            groups.setdefault("a", []).extend(unassigned_leaf_ids)

        endorsed: list[list[int]] = []
        seen_groups: set[tuple[int, ...]] = set()
        for group_items in groups.values():
            unique_items = sorted(set(group_items))
            group_tuple = tuple(unique_items)
            if group_tuple not in seen_groups:
                endorsed.append(unique_items)
                seen_groups.add(group_tuple)

        return Judgement(
            user_id=user_id,
            shown=self.cluster_id,
            endorsed=endorsed,
        )


def create_display_dataframe(
    df: pl.DataFrame, source_configs: list[tuple[str, SourceConfig]]
) -> pl.DataFrame:
    """Create enhanced display DataFrame for flexible view rendering."""
    leaf_ids = (
        df.select("leaf").to_series().to_list()
        if "leaf" in df.columns
        else list(range(len(df)))
    )

    rows: list[dict[str, object]] = []
    for record_idx, record in enumerate(df.iter_rows(named=True)):
        leaf_id = leaf_ids[record_idx]
        for source_name, source_config in source_configs:
            for field in source_config.index_fields:
                qualified_field = source_config.f(source_name, field.name)
                if qualified_field not in record:
                    continue

                value = record[qualified_field]
                if value is None:
                    continue

                str_value = str(value).strip()
                if not str_value:
                    continue

                rows.append(
                    {
                        "field_name": field.name,
                        "record_index": record_idx,
                        "value": str_value,
                        "leaf_id": leaf_id,
                    }
                )

    if rows:
        return pl.DataFrame(rows)

    return pl.DataFrame(
        {
            "field_name": [],
            "record_index": [],
            "value": [],
            "leaf_id": [],
        },
        schema={
            "field_name": pl.String,
            "record_index": pl.Int32,
            "value": pl.String,
            "leaf_id": pl.Int32,
        },
    )


class DeduplicationResult:
    """Result of column deduplication analysis."""

    def __init__(
        self,
        duplicate_groups: list[list[int]],
        display_columns: list[int],
        leaf_to_display_mapping: dict[int, int],
    ) -> None:
        """Initialise deduplication metadata for mapping leaf columns to displays."""
        self.duplicate_groups = duplicate_groups
        self.display_columns = display_columns
        self.leaf_to_display_mapping = leaf_to_display_mapping


def deduplicate_columns(display_df: pl.DataFrame) -> DeduplicationResult:
    """Analyse columns for duplicates and create deduplication mapping."""
    if display_df.is_empty():
        return DeduplicationResult([], [], {})

    leaf_ids = sorted(display_df["leaf_id"].unique().to_list())
    if not leaf_ids:
        return DeduplicationResult([], [], {})

    column_signatures: dict[int, tuple[tuple[str, str], ...]] = {}
    for leaf_id in leaf_ids:
        column_data = (
            display_df.filter(pl.col("leaf_id") == leaf_id)
            .sort(["field_name", "record_index"])
            .select(["field_name", "value"])
        )
        signature = tuple(
            zip(column_data["field_name"], column_data["value"], strict=True)
        )
        column_signatures[leaf_id] = signature

    signature_to_leaves: dict[tuple[tuple[str, str], ...], list[int]] = {}
    for leaf_id, signature in column_signatures.items():
        signature_to_leaves.setdefault(signature, []).append(leaf_id)

    duplicate_groups: list[list[int]] = []
    display_columns: list[int] = []
    leaf_to_display_mapping: dict[int, int] = {}

    for display_col_index, leaf_group in enumerate(signature_to_leaves.values()):
        leaf_group = sorted(leaf_group)
        representative_leaf = leaf_group[0]
        duplicate_groups.append(leaf_group)
        display_columns.append(representative_leaf)
        for leaf_id in leaf_group:
            leaf_to_display_mapping[leaf_id] = display_col_index

    return DeduplicationResult(
        duplicate_groups=duplicate_groups,
        display_columns=display_columns,
        leaf_to_display_mapping=leaf_to_display_mapping,
    )


def create_evaluation_item(
    df: pl.DataFrame, source_configs: list[tuple[str, SourceConfig]], cluster_id: int
) -> EvaluationItem:
    """Create a complete EvaluationItem with deduplication."""
    display_dataframe = create_display_dataframe(df, source_configs)
    dedup_result = deduplicate_columns(display_dataframe)

    return EvaluationItem(
        cluster_id=cluster_id,
        dataframe=df,
        display_dataframe=display_dataframe,
        duplicate_groups=dedup_result.duplicate_groups,
        display_columns=dedup_result.display_columns,
        leaf_to_display_mapping=dedup_result.leaf_to_display_mapping,
        assignments={},
    )


def get_samples(
    n: int,
    resolution: ModelResolutionPath,
    user_id: int,
    dag: DAG,
) -> dict[int, EvaluationItem]:
    """Retrieve samples enriched with source data, as EvaluationItems."""
    samples: pl.DataFrame = cast(
        pl.DataFrame,
        pl.from_arrow(
            _handler.sample_for_eval(n=n, resolution=resolution, user_id=user_id)
        ),
    )

    if not len(samples):
        return {}

    results_by_source: list[pl.DataFrame] = []
    source_configs: list[tuple[str, SourceConfig]] = []

    for source_resolution in samples["source"].unique():
        try:
            source = dag.get_source(source_resolution)
        except ValueError as exc:  # pragma: no cover - defensive path
            raise MatchboxSourceTableError(
                f"Source '{source_resolution}' not found in DAG. "
                "Ensure DAG was loaded with all resolutions."
            ) from exc

        source_configs.append((source_resolution, source.config))

        samples_by_source = samples.filter(pl.col("source") == source_resolution)
        keys_by_source = samples_by_source["key"].to_list()

        try:
            source_data = pl.concat(
                source.fetch(batch_size=10_000, qualify_names=True, keys=keys_by_source)
            )
        except OperationalError as exc:
            raise MatchboxSourceTableError(
                f"Could not query source '{source_resolution}' from warehouse. "
                "Check warehouse connection and ensure source table exists."
            ) from exc

        samples_and_source = samples_by_source.join(
            source_data, left_on="key", right_on=source.qualified_key
        )
        desired_columns = ["root", "leaf", "key"] + source.qualified_index_fields
        results_by_source.append(samples_and_source[desired_columns])

    if not results_by_source:
        return {}

    all_results: pl.DataFrame = pl.concat(results_by_source, how="diagonal")

    results_by_root: dict[int, EvaluationItem] = {}
    for root in all_results["root"].unique():
        cluster_df = all_results.filter(pl.col("root") == root).drop("root")
        evaluation_item = create_evaluation_item(cluster_df, source_configs, root)
        results_by_root[root] = evaluation_item

    return results_by_root


class EvalData:
    """Object which caches evaluation data to measure model performance."""

    def __init__(self) -> None:
        """Download judgement and expansion data used to compute evaluation metrics."""
        self.judgements, self.expansion = _handler.download_eval_data()

    def precision_recall(
        self, results: Results, threshold: float
    ) -> tuple[float, float]:
        """Compute precision and recall for a given Results object."""
        if not len(results.clusters):
            raise ValueError("No clusters suggested by these results.")

        threshold = int(threshold * 100)
        root_leaf = results.root_leaf().rename({"root_id": "root", "leaf_id": "leaf"})
        values = precision_recall([root_leaf], self.judgements, self.expansion)[0]
        return values[0], values[1]


def compare_models(resolutions: list[ModelResolutionPath]) -> ModelComparison:
    """Compare metrics of models based on cached evaluation data."""
    return _handler.compare_models(resolutions)


__all__ = [
    "DeduplicationResult",
    "EvaluationItem",
    "create_display_dataframe",
    "create_evaluation_item",
    "deduplicate_columns",
    "EvalData",
    "get_samples",
    "compare_models",
    "ModelComparison",
    "precision_recall",
]
