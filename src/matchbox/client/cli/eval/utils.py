"""Collection of client-side functions in aid of model evaluation."""

import warnings
from typing import Any

import polars as pl
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.results import Results
from matchbox.common.eval import (
    Judgement,
    ModelComparison,
    PrecisionRecall,
    precision_recall,
)
from matchbox.common.exceptions import MatchboxSourceTableError
from matchbox.common.graph import DEFAULT_RESOLUTION, ModelResolutionName
from matchbox.common.logging import logger
from matchbox.common.sources import SourceConfig

try:
    from pydantic import BaseModel, computed_field
except ImportError:
    BaseModel = object

    def computed_field():
        """Fallback decorator when Pydantic is not available."""

        def decorator(func):
            return func

        return decorator


class EvaluationItem(BaseModel):
    """A cluster awaiting evaluation, with processed field data."""

    model_config = {"arbitrary_types_allowed": True}

    cluster_id: int
    dataframe: pl.DataFrame
    display_dataframe: pl.DataFrame  # Enhanced DataFrame for flexible rendering
    field_names: list[str]  # Pre-processed field names for display
    data_matrix: list[list[str]]  # Pre-processed data matrix
    leaf_ids: list[int]  # Pre-processed leaf IDs
    assignments: dict[int, str] = {}  # column_index -> group_letter

    @computed_field
    @property
    def total_columns(self) -> int:
        """Total number of columns/leaves in this cluster."""
        return len(self.leaf_ids)

    @computed_field
    @property
    def is_painted(self) -> bool:
        """True only if ALL columns have been assigned to groups."""
        return len(self.assignments) == self.total_columns

    def to_judgement(self, user_id: int) -> Judgement:
        """Convert assignments to Judgement format for submission."""
        groups = {}
        # Group leaf IDs by assignment
        for col_index, group in self.assignments.items():
            if group not in groups:
                groups[group] = []
            if col_index < len(self.leaf_ids):
                groups[group].append(self.leaf_ids[col_index])

        # Unassigned columns default to group 'a'
        assigned_cols = set(self.assignments.keys())
        unassigned = [
            self.leaf_ids[i]
            for i in range(len(self.leaf_ids))
            if i not in assigned_cols
        ]
        if unassigned:
            groups.setdefault("a", []).extend(unassigned)

        # Convert to endorsed format, ensuring no duplicate groups
        endorsed = []
        seen_groups = set()
        for group_items in groups.values():
            # Remove duplicates within the group and sort for consistent comparison
            unique_items = sorted(set(group_items))
            group_tuple = tuple(unique_items)
            # Only add if we haven't seen this exact group before
            if group_tuple not in seen_groups:
                endorsed.append(unique_items)
                seen_groups.add(group_tuple)

        return Judgement(
            user_id=user_id,
            shown=self.cluster_id,
            endorsed=endorsed,
        )


def create_display_dataframe(
    df: pl.DataFrame, source_configs: list[SourceConfig]
) -> pl.DataFrame:
    """Create enhanced display DataFrame for flexible view rendering.

    Args:
        df: DataFrame with records as rows and qualified fields as columns
        source_configs: List of SourceConfig objects that generated the qualified fields

    Returns:
        DataFrame with columns: field_name, source_name, record_index, value, leaf_id
    """
    # Get leaf IDs for records
    leaf_ids = (
        df.select("leaf").to_series().to_list()
        if "leaf" in df.columns
        else list(range(len(df)))
    )

    rows = []

    # Iterate through each record in the DataFrame
    for record_idx, record in enumerate(df.iter_rows(named=True)):
        leaf_id = leaf_ids[record_idx]

        # For each source config, extract its fields
        for source_config in source_configs:
            source_name = source_config.name

            for field in source_config.index_fields:
                qualified_field = source_config.f(field.name)
                unqualified_field = field.name

                # Only include fields that exist in the DataFrame
                if qualified_field in record:
                    value = record[qualified_field]
                    # Only include non-null, non-empty values
                    if value is not None:
                        str_value = str(value).strip()
                        if str_value:  # Only add non-empty strings
                            rows.append(
                                {
                                    "field_name": unqualified_field,
                                    "source_name": source_name,
                                    "record_index": record_idx,
                                    "value": str_value,
                                    "leaf_id": leaf_id,
                                }
                            )

    # Create DataFrame from collected rows
    if rows:
        return pl.DataFrame(rows)
    else:
        # Return empty DataFrame with correct schema
        return pl.DataFrame(
            {
                "field_name": [],
                "source_name": [],
                "record_index": [],
                "value": [],
                "leaf_id": [],
            },
            schema={
                "field_name": pl.String,
                "source_name": pl.String,
                "record_index": pl.Int32,
                "value": pl.String,
                "leaf_id": pl.Int32,
            },
        )


def dataframe_to_legacy_format(
    display_df: pl.DataFrame, num_records: int
) -> tuple[list[str], list[list[str]]]:
    """Convert display DataFrame to legacy field_names and data_matrix format.

    Args:
        display_df: Enhanced display DataFrame
        num_records: Total number of records (for padding empty values)

    Returns:
        Tuple of (field_names, data_matrix) in legacy format
    """
    field_names = []
    data_matrix = []

    if display_df.is_empty():
        return field_names, data_matrix

    # Group by field_name to recreate the current grouped structure
    field_groups = display_df.group_by("field_name").agg(
        [pl.col("source_name"), pl.col("record_index"), pl.col("value")]
    )

    for field_row in field_groups.iter_rows(named=True):
        field_name = field_row["field_name"]
        source_names = field_row["source_name"]
        record_indices = field_row["record_index"]
        values = field_row["value"]

        # Add separator if not the first group
        if field_names:
            field_names.append("---")
            data_matrix.append([""] * num_records)

        # Group by source within this field
        source_data = {}
        for source, record_idx, value in zip(
            source_names, record_indices, values, strict=True
        ):
            if source not in source_data:
                source_data[source] = [""] * num_records
            source_data[source][record_idx] = value

        # Add a row for each source's version of this field
        for source_name, row_values in source_data.items():
            field_names.append(f"{field_name} ({source_name})")
            data_matrix.append(row_values)

    return field_names, data_matrix


def create_processed_comparison_data(
    df: pl.DataFrame, source_configs: list[SourceConfig]
) -> tuple[pl.DataFrame, list[str], list[list[str]], list[int]]:
    """Create comparison data with enhanced DataFrame and legacy formats.

    Args:
        df: DataFrame with records as rows and qualified fields as columns
        source_configs: List of SourceConfig objects that generated the qualified fields

    Returns:
        Tuple of:
        - display_dataframe: Enhanced DataFrame for flexible rendering
        - field_names: List of field names (row headers) - legacy format
        - data_matrix: List of rows, each containing values for all records
        - leaf_ids: List of leaf IDs for each record (column)
    """
    # Get leaf IDs for records (these become our column identifiers)
    leaf_ids = (
        df.select("leaf").to_series().to_list()
        if "leaf" in df.columns
        else list(range(len(df)))
    )

    # Create the enhanced display DataFrame
    display_dataframe = create_display_dataframe(df, source_configs)

    # Generate backward-compatible field_names and data_matrix
    field_names, data_matrix = dataframe_to_legacy_format(display_dataframe, len(df))

    return display_dataframe, field_names, data_matrix, leaf_ids


def get_samples(
    n: int,
    user_id: int,
    resolution: ModelResolutionName | None = None,
    clients: dict[str, Any] | None = None,
    use_default_client: bool = False,
) -> dict[int, EvaluationItem]:
    """Retrieve samples enriched with source data, grouped by resolution cluster.

    Args:
        n: Number of clusters to sample
        user_id: ID of the user requesting the samples
        resolution: Model resolution proposing the clusters. If not set, will
            use a default resolution.
        clients: Dictionary from location names to valid client for each.
            Locations whose name is missing from the dictionary will be skipped.
        use_default_client: Whether to use for all unset location clients
            a SQLAlchemy engine for the default warehouse set in the environment
            variable `MB__CLIENT__DEFAULT_WAREHOUSE`.

    Returns:
        Dictionary of cluster ID to EvaluationItem with processed field data

    Raises:
        MatchboxSourceTableError: If a source cannot be queried from a location using
            provided or default clients.
    """
    if not resolution:
        resolution = DEFAULT_RESOLUTION

    if not clients:
        clients = {}

    default_client = None
    if use_default_client:
        if default_clients_uri := settings.default_warehouse:
            default_client = create_engine(default_clients_uri)
            logger.warning("Using default engine")
        else:
            raise ValueError("`MB__CLIENT__DEFAULT_WAREHOUSE` is unset")

    try:
        samples: pl.DataFrame = pl.from_arrow(
            _handler.sample_for_eval(n=n, resolution=resolution, user_id=user_id)
        )

        if not len(samples):
            return {}

        results_by_source = []
        source_configs = []
        for source_resolution in samples["source"].unique():
            source_config = _handler.get_source_config(source_resolution)
            source_configs.append(source_config)
            location_name = source_config.location.name
            if location_name in clients:
                source_config.location.add_client(client=clients[location_name])
            elif default_client:
                source_config.location.add_client(client=default_client)
            else:
                warnings.warn(
                    f"Skipping {source_resolution}, incompatible with given client.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            samples_by_source = samples.filter(pl.col("source") == source_resolution)
            keys_by_source = samples_by_source["key"].to_list()

            try:
                source_data = pl.concat(
                    source_config.query(
                        batch_size=10_000, qualify_names=True, keys=keys_by_source
                    )
                )
            except OperationalError as e:
                raise MatchboxSourceTableError(
                    "Could not find source using given client."
                ) from e

            samples_and_source = samples_by_source.join(
                source_data, left_on="key", right_on=source_config.qualified_key
            )
            desired_columns = ["root", "leaf", "key"] + source_config.qualified_fields
            results_by_source.append(samples_and_source[desired_columns])

        if not len(results_by_source):
            return {}

        all_results: pl.DataFrame = pl.concat(results_by_source, how="diagonal")

        results_by_root = {}
        for root in all_results["root"].unique():
            cluster_df = all_results.filter(pl.col("root") == root).drop("root")

            # Process field data using SourceConfig information
            (
                display_dataframe,
                field_names,
                data_matrix,
                leaf_ids,
            ) = create_processed_comparison_data(cluster_df, source_configs)

            # Create EvaluationItem with processed data
            evaluation_item = EvaluationItem(
                cluster_id=int(root),
                dataframe=cluster_df,
                display_dataframe=display_dataframe,
                field_names=field_names,
                data_matrix=data_matrix,
                leaf_ids=leaf_ids,
                assignments={},
            )

            results_by_root[int(root)] = evaluation_item

        return results_by_root
    finally:
        # Always dispose of the default client if we created one
        if default_client:
            default_client.dispose()


class EvalData:
    """Object which caches evaluation data to measure performance of models."""

    def __init__(self):
        """Initialise evaluation data from resolution name."""
        self.judgements, self.expansion = _handler.download_eval_data()

    def precision_recall(self, results: Results, threshold: float) -> PrecisionRecall:
        """Computes precision and recall at one threshold."""
        if not len(results.clusters):
            raise ValueError("No clusters suggested by these results.")

        threshold = int(threshold * 100)

        root_leaf = (
            results.root_leaf()
            .rename({"root_id": "root", "leaf_id": "leaf"})
            .to_arrow()
        )
        return precision_recall([root_leaf], self.judgements, self.expansion)[0]

    def pr_curve(self, results: Results) -> Figure:
        """Computes precision and recall for each threshold in results."""
        all_p = []
        all_r = []

        probs = pl.from_arrow(results.probabilities)
        thresholds = probs.select("probability").unique().to_series()
        for i, t in enumerate(sorted(thresholds)):
            float_thresh = t / 100
            p, r = self.precision_recall(results=results, threshold=float_thresh)
            all_p.append(p)
            all_r.append(r)
            plt.annotate(float_thresh, (all_r[i], all_p[i]))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(all_r, all_p, marker="o")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        ax.set_title("Precision-Recall Curve")
        ax.grid()

        return fig


def compare_models(resolutions: list[ModelResolutionName]) -> ModelComparison:
    """Compare metrics of models based on evaluation data.

    Args:
        resolutions: List of names of model resolutions to be compared.

    Returns:
        A model comparison object, listing metrics for each model.
    """
    return _handler.compare_models(resolutions)
