"""Collection of client-side functions in aid of model evaluation."""

import logging
import warnings
from contextlib import contextmanager
from itertools import combinations
from typing import Any

import polars as pl
import pyarrow as pa
from pydantic import BaseModel, computed_field
from sqlalchemy.exc import OperationalError

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.dags import DAG
from matchbox.client.results import Results
from matchbox.client.sources import Location, Source
from matchbox.common.dtos import (
    ModelResolutionPath,
    ResolutionPath,
    ResolutionType,
    SourceConfig,
)
from matchbox.common.eval import (
    Judgement,
    ModelComparison,
    Pair,
    Pairs,
    process_judgements,
    wilson_confidence_interval,
)
from matchbox.common.exceptions import MatchboxSourceTableError
from matchbox.common.transform import DisjointSet


@contextmanager
def temp_warehouse(warehouse: str | None):
    """Temporarily set the default warehouse in settings."""
    original_warehouse = getattr(settings, "default_warehouse", None)
    if warehouse:
        settings.default_warehouse = warehouse
    try:
        yield
    finally:
        if warehouse:
            if original_warehouse:
                settings.default_warehouse = original_warehouse
            elif hasattr(settings, "default_warehouse"):
                delattr(settings, "default_warehouse")


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
        groups = {}

        # Expand display column assignments to all underlying leaf IDs
        for display_col_index, group in self.assignments.items():
            if group not in groups:
                groups[group] = []

            # Get all leaf IDs for this display column (including duplicates)
            if display_col_index < len(self.duplicate_groups):
                duplicate_group = self.duplicate_groups[display_col_index]
                groups[group].extend(duplicate_group)

        # Handle unassigned display columns - expand to all their leaf IDs
        assigned_display_cols = set(self.assignments.keys())
        unassigned_leaf_ids = []

        for display_col_index in range(len(self.duplicate_groups)):
            if display_col_index not in assigned_display_cols:
                unassigned_leaf_ids.extend(self.duplicate_groups[display_col_index])

        if unassigned_leaf_ids:
            groups.setdefault("a", []).extend(unassigned_leaf_ids)

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
    df: pl.DataFrame, source_configs: list[tuple[str, SourceConfig]]
) -> pl.DataFrame:
    """Create enhanced display DataFrame for flexible view rendering.

    Args:
        df: DataFrame with records as rows and qualified fields as columns
        source_configs: List of (source_name, SourceConfig) tuples

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
        for source_name, source_config in source_configs:
            for field in source_config.index_fields:
                qualified_field = source_config.f(source_name, field.name)
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


class DeduplicationResult:
    """Result of column deduplication analysis."""

    def __init__(
        self,
        duplicate_groups: list[list[int]],
        display_columns: list[int],
        leaf_to_display_mapping: dict[int, int],
    ):
        """Initialise deduplication result."""
        self.duplicate_groups = duplicate_groups
        self.display_columns = display_columns
        self.leaf_to_display_mapping = leaf_to_display_mapping


def deduplicate_columns(display_df: pl.DataFrame) -> DeduplicationResult:
    """Analyze columns for duplicates and create deduplication mapping.

    Args:
        display_df: Enhanced display DataFrame

    Returns:
        DeduplicationResult with duplicate groupings and mappings
    """
    if display_df.is_empty():
        return DeduplicationResult([], [], {})

    # Get all unique leaf_ids (columns)
    leaf_ids = sorted(display_df["leaf_id"].unique().to_list())

    if not leaf_ids:
        return DeduplicationResult([], [], {})

    # Create column signatures by hashing all field values for each column
    column_signatures = {}

    for leaf_id in leaf_ids:
        # Get all data for this column across all fields
        column_data = (
            display_df.filter(pl.col("leaf_id") == leaf_id)
            .sort("field_name")  # Consistent ordering
            .select(["field_name", "value"])
        )

        # Create a signature from all field-value pairs
        if not column_data.is_empty():
            # Convert to sorted list of (field, value) pairs for consistent hashing
            field_value_pairs = [
                (row["field_name"], row["value"])
                for row in column_data.iter_rows(named=True)
            ]
            signature = tuple(sorted(field_value_pairs))
        else:
            signature = ()  # Empty column

        column_signatures[leaf_id] = signature

    # Group columns by identical signatures
    signature_to_leaves = {}
    for leaf_id, signature in column_signatures.items():
        if signature not in signature_to_leaves:
            signature_to_leaves[signature] = []
        signature_to_leaves[signature].append(leaf_id)

    # Build deduplication structures
    duplicate_groups = []
    display_columns = []
    leaf_to_display_mapping = {}

    for display_col_index, leaf_group in enumerate(signature_to_leaves.values()):
        # Sort leaf IDs for consistent ordering
        leaf_group = sorted(leaf_group)

        # First leaf ID in group becomes the display representative
        representative_leaf = leaf_group[0]

        duplicate_groups.append(leaf_group)
        display_columns.append(representative_leaf)

        # Map all leaves in this group to the same display column index
        for leaf_id in leaf_group:
            leaf_to_display_mapping[leaf_id] = display_col_index

    return DeduplicationResult(
        duplicate_groups, display_columns, leaf_to_display_mapping
    )


def create_evaluation_item(
    df: pl.DataFrame, source_configs: list[tuple[str, SourceConfig]], cluster_id: int
) -> EvaluationItem:
    """Create a complete EvaluationItem with deduplication.

    Args:
        df: DataFrame with records as rows and qualified fields as columns
        source_configs: List of SourceConfig objects
        cluster_id: The cluster ID for this evaluation item

    Returns:
        Complete EvaluationItem with deduplication applied
    """
    # Create enhanced display DataFrame
    display_dataframe = create_display_dataframe(df, source_configs)

    # Perform deduplication analysis
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
    clients: dict[str, Any] | None = None,
    default_client: Any | None = None,
) -> dict[int, EvaluationItem]:
    """Retrieve samples enriched with source data, as EvaluationItems.

    Args:
        n: Number of clusters to sample
        resolution: Model resolution to retrieve samples for
        user_id: ID of the user requesting the samples
        clients: Dictionary from location names to valid client for each.
            Locations whose name is missing from the dictionary will be skipped,
            unless a default client is provided.
        default_client: Fallback client to use for all sources.

    Returns:
        Dictionary of cluster ID to EvaluationItem

    Raises:
        MatchboxSourceTableError: If a source cannot be queried from a location using
            provided or default clients.
    """
    if not clients:
        clients = {}

    # Create temporary DAG for Source reconstruction
    temp_dag = DAG(name=str(resolution.collection))
    temp_dag._run = resolution.run

    samples: pl.DataFrame = pl.from_arrow(
        _handler.sample_for_eval(n=n, resolution=resolution, user_id=user_id)
    )

    if not len(samples):
        return {}

    results_by_source = []
    source_configs: list[tuple[str, SourceConfig]] = []

    for source_resolution in samples["source"].unique():
        source_resolution_obj = _handler.get_resolution(
            path=ResolutionPath(
                name=source_resolution, collection=temp_dag.name, run=temp_dag.run
            ),
            validate_type=ResolutionType.SOURCE,
        )
        location_name = source_resolution_obj.config.location_config.name

        if location_name in clients:
            client = clients[location_name]
        elif default_client:
            client = default_client
        else:
            warnings.warn(
                f"Skipping {source_resolution}, incompatible with given client.",
                UserWarning,
                stacklevel=2,
            )
            continue

        location = Location.from_config(
            source_resolution_obj.config.location_config, client=client
        )
        source = Source.from_resolution(
            resolution=source_resolution_obj,
            resolution_name=source_resolution,
            dag=temp_dag,
            location=location,
        )

        # Store source config for later EvaluationItem creation
        source_configs.append((source_resolution, source_resolution_obj.config))

        samples_by_source = samples.filter(pl.col("source") == source_resolution)
        keys_by_source = samples_by_source["key"].to_list()

        try:
            source_data = pl.concat(
                source.fetch(batch_size=10_000, qualify_names=True, keys=keys_by_source)
            )
        except OperationalError as e:
            raise MatchboxSourceTableError(
                "Could not find source using given client."
            ) from e

        samples_and_source = samples_by_source.join(
            source_data, left_on="key", right_on=source.qualified_key
        )
        desired_columns = ["root", "leaf", "key"] + source.qualified_index_fields
        results_by_source.append(samples_and_source[desired_columns])

    if not results_by_source:
        return {}

    all_results: pl.DataFrame = pl.concat(results_by_source, how="diagonal")

    # Convert to EvaluationItems
    results_by_root: dict[int, EvaluationItem] = {}
    for root in all_results["root"].unique():
        cluster_df = all_results.filter(pl.col("root") == root).drop("root")
        evaluation_item = create_evaluation_item(cluster_df, source_configs, root)
        results_by_root[root] = evaluation_item

    return results_by_root


class IncrementalState:
    """Manages incremental state for optimised PR curve calculation using Union-Find."""

    def __init__(
        self,
        validation_pairs: Pairs,
        validation_counts: dict[Pair, float],
        validation_leaves: set[int],
    ):
        """Initialise with preprocessed validation data."""
        self.validation_pairs = validation_pairs
        self.validation_counts = validation_counts
        self.validation_leaves = validation_leaves

        # Union-Find for tracking connected components
        self.disjoint_set: DisjointSet[int] = DisjointSet()

        # Track current model pairs at this threshold
        self.current_model_pairs: Pairs = set()

        # Add all validation leaves to disjoint set
        for leaf in validation_leaves:
            self.disjoint_set.add(leaf)

    def add_threshold_edges(
        self, threshold: float, probabilities: pa.Table, model_root_leaf: pa.Table
    ):
        """Add edges at the given threshold and update model pairs."""
        if probabilities is None:
            # No probabilities - include all model pairs
            self._add_all_model_pairs(model_root_leaf)
            return

        # Convert threshold back to percentage for comparison with probabilities
        threshold_pct = int(threshold * 100)

        # Get edges at this threshold
        probs_df = pl.from_arrow(probabilities)
        threshold_edges = probs_df.filter(pl.col("probability") >= threshold_pct)

        if threshold_edges.is_empty():
            return

        # Add edges to Union-Find and update model pairs
        for row in threshold_edges.iter_rows(named=True):
            left_id = row["left_id"]
            right_id = row["right_id"]

            # Only consider edges between validation leaves
            if left_id in self.validation_leaves and right_id in self.validation_leaves:
                # Add to Union-Find
                self.disjoint_set.union(left_id, right_id)

                # Add pair to current model pairs (sorted for consistency)
                pair = (min(left_id, right_id), max(left_id, right_id))
                if pair in self.validation_counts and self.validation_counts[pair] != 0:
                    self.current_model_pairs.add(pair)

    def _add_all_model_pairs(self, model_root_leaf: pa.Table):
        """Add all model pairs when no probability filtering is needed."""
        # Convert model clusters to pairs
        clusters = (
            pl.from_arrow(model_root_leaf)
            .group_by("root")
            .agg(pl.col("leaf").alias("leaves"))
            .select("leaves")
            .to_series()
            .to_list()
        )

        for cluster_leaves in clusters:
            # Generate all pairs within this cluster
            for left, right in combinations(sorted(cluster_leaves), r=2):
                if left in self.validation_leaves and right in self.validation_leaves:
                    # Add to Union-Find
                    self.disjoint_set.union(left, right)

                    # Add to model pairs if it has judgements
                    pair = (left, right)
                    if (
                        pair in self.validation_counts
                        and self.validation_counts[pair] != 0
                    ):
                        self.current_model_pairs.add(pair)

    def calculate_current_metrics(self) -> tuple[float, float, float, float]:
        """Calculate precision, recall, and confidence intervals for current state."""
        # Filter validation pairs to only include positively judged ones
        positive_validation_pairs = {
            pair
            for pair in self.validation_pairs
            if pair in self.validation_counts and self.validation_counts[pair] > 0
        }

        # Calculate true positives (intersection of model and positive validation pairs)
        true_positive_pairs = self.current_model_pairs & positive_validation_pairs

        # Calculate precision and recall
        precision = (
            len(true_positive_pairs) / len(self.current_model_pairs)
            if self.current_model_pairs
            else 0.0
        )
        recall = (
            len(true_positive_pairs) / len(positive_validation_pairs)
            if positive_validation_pairs
            else 0.0
        )

        # Calculate Wilson confidence intervals
        precision_ci = wilson_confidence_interval(
            len(true_positive_pairs), len(self.current_model_pairs)
        )
        recall_ci = wilson_confidence_interval(
            len(true_positive_pairs), len(positive_validation_pairs)
        )

        return precision, recall, precision_ci, recall_ci


class EvalData:
    """Object which caches evaluation data to measure performance of models."""

    def __init__(
        self,
        root_leaf: pa.Table,
        thresholds: list[float],
        probabilities: pa.Table = None,
    ):
        """Initialise evaluation data with root/leaf mapping and thresholds.

        Args:
            root_leaf: PyArrow table with 'root' and 'leaf' columns
            thresholds: List of probability thresholds (as floats 0.0-1.0)
            probabilities: Optional PyArrow table with probability data
        """
        self.root_leaf = root_leaf
        self.thresholds = thresholds
        self.probabilities = probabilities
        self.judgements, self.expansion = _handler.download_eval_data()

        # Cache expensive judgement processing for reuse (if judgements exist)
        if len(self.judgements) > 0:
            (
                self._validation_pairs,
                self._validation_counts,
                self._validation_leaves,
            ) = process_judgements(
                pl.from_arrow(self.judgements), pl.from_arrow(self.expansion)
            )
        else:
            # Handle empty judgements gracefully
            self._validation_pairs = set()
            self._validation_counts = {}
            self._validation_leaves = set()

    @classmethod
    def from_results(cls, results: Results) -> "EvalData":
        """Create EvalData from a Results object (existing functionality).

        Args:
            results: Results object containing clusters and probabilities

        Returns:
            EvalData instance ready for precision/recall calculations
        """
        # Extract root_leaf mapping from results
        root_leaf = (
            results.root_leaf()
            .rename({"root_id": "root", "leaf_id": "leaf"})
            .to_arrow()
        )

        # Extract thresholds from probabilities
        probs = pl.from_arrow(results.probabilities)
        thresholds = sorted(probs.select("probability").unique().to_series().to_list())
        thresholds = [t / 100 for t in thresholds]  # Convert to 0.0-1.0 range

        return cls(root_leaf, thresholds, results.probabilities)

    @classmethod
    def from_resolution(cls, resolution: ModelResolutionPath) -> "EvalData":
        """Create EvalData from a model resolution (new functionality).

        Args:
            resolution: Model resolution name to build data from

        Returns:
            EvalData instance ready for precision/recall calculations
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting EvalData.from_resolution() for resolution: {resolution}")

        # Get model configuration
        logger.info(f"Fetching model config for resolution: {resolution}")
        model_resolution = _handler.get_resolution(
            path=resolution, validate_type=ResolutionType.MODEL
        )
        model_config = model_resolution.config
        if not model_config:
            logger.error(f"Model {resolution} not found")
            raise ValueError(f"Model {resolution} not found")

        # Extract source resolutions from query configs
        left_sources = model_config.left_query.source_resolutions
        right_sources = (
            model_config.right_query.source_resolutions
            if model_config.right_query
            else None
        )
        logger.info(
            f"Model config retrieved: left_sources={left_sources}, "
            f"right_sources={right_sources}"
        )

        # Get model probabilities/results
        logger.info(f"Fetching model results for resolution: {resolution}")
        probabilities = _handler.get_results(resolution)
        probs_df = pl.from_arrow(probabilities)
        logger.info(f"Model results retrieved: {len(probs_df)} probability records")

        # Extract thresholds from probabilities
        logger.info("Extracting thresholds from probabilities")
        thresholds = sorted(
            probs_df.select("probability").unique().to_series().to_list()
        )
        logger.info(f"Raw thresholds extracted: {thresholds}")
        thresholds = [t / 100 for t in thresholds]  # Convert to 0.0-1.0 range
        logger.info(f"Normalized thresholds: {thresholds}")

        # Handle both deduper and linker models
        if model_config.right_query is None:
            logger.info("Processing deduper model (single source)")
            # Deduper model - left_query contains the only source
            left_source = model_config.left_query.source_resolutions[0]
            logger.info(f"Querying source data for: {left_source}")
            source_data = _handler.query(
                source=left_source,
                resolution=left_source,
                return_leaf_id=True,
            )
            source_df = pl.from_arrow(source_data)
            logger.info(f"Source data retrieved: {len(source_df)} records")

            # Create root_leaf mapping from probabilities and leaf_id
            logger.info("Creating id_to_leaf mapping")
            id_to_leaf = dict(
                zip(
                    source_df["id"].to_list(),
                    source_df["leaf_id"].to_list(),
                    strict=False,
                )
            )
            logger.info(f"Created {len(id_to_leaf)} id->leaf mappings")

            # Map left_id and right_id to their leaf_ids
            logger.info("Building root_leaf data from probabilities")
            root_leaf_data = []
            for row in probs_df.iter_rows(named=True):
                left_leaf = id_to_leaf.get(row["left_id"])
                right_leaf = id_to_leaf.get(row["right_id"])
                if left_leaf is not None and right_leaf is not None:
                    root_leaf_data.append({"root": left_leaf, "leaf": right_leaf})

            root_leaf_df = pl.DataFrame(root_leaf_data).unique()
            logger.info(
                f"Created root_leaf dataframe with {len(root_leaf_df)} unique mappings"
            )
        else:
            logger.info("Processing linker model (two sources)")
            # Linker model - has both left and right queries with sources
            left_source = model_config.left_query.source_resolutions[0]
            right_source = model_config.right_query.source_resolutions[0]
            logger.info(f"Querying left source data for: {left_source}")
            left_data = _handler.query(
                source=left_source,
                resolution=left_source,
                return_leaf_id=True,
            )
            logger.info(f"Querying right source data for: {right_source}")
            right_data = _handler.query(
                source=right_source,
                resolution=right_source,
                return_leaf_id=True,
            )

            # Build root_leaf mapping manually
            left_df = pl.from_arrow(left_data)
            right_df = pl.from_arrow(right_data)
            logger.info(
                f"Left data: {len(left_df)} records, "
                f"Right data: {len(right_df)} records"
            )

            # Create leaf_id lookup tables
            logger.info("Creating lookup tables")
            left_lookup = left_df.select(["id", "leaf_id"]).rename(
                {"leaf_id": "left_leaf"}
            )
            right_lookup = right_df.select(["id", "leaf_id"]).rename(
                {"leaf_id": "right_leaf"}
            )

            # Join probabilities with leaf lookups to build root_leaf mapping
            logger.info("Building root_leaf mapping via joins")
            root_leaf_df = (
                probs_df.join(left_lookup, left_on="left_id", right_on="id", how="left")
                .join(right_lookup, left_on="right_id", right_on="id", how="left")
                .select(
                    [
                        pl.col("left_leaf").alias("root"),
                        pl.col("right_leaf").alias("leaf"),
                    ]
                )
                .unique()
            )
            logger.info(
                f"Created root_leaf dataframe with {len(root_leaf_df)} unique mappings"
            )

        logger.info("Converting to Arrow format and creating EvalData instance")
        root_leaf = root_leaf_df.to_arrow()
        logger.info(f"Final root_leaf Arrow table: {len(root_leaf)} records")
        result = cls(root_leaf, thresholds, probabilities)
        logger.info("EvalData.from_resolution() completed successfully")
        return result

    def refresh_judgements(self) -> None:
        """Refresh judgement data with latest submissions from the server."""
        self.judgements, self.expansion = _handler.download_eval_data()

        # Refresh cached judgement processing
        if len(self.judgements) > 0:
            (
                self._validation_pairs,
                self._validation_counts,
                self._validation_leaves,
            ) = process_judgements(
                pl.from_arrow(self.judgements), pl.from_arrow(self.expansion)
            )
        else:
            # Handle empty judgements gracefully
            self._validation_pairs = set()
            self._validation_counts = {}
            self._validation_leaves = set()

    def precision_recall(
        self, thresholds: list[float] | None = None
    ) -> list[tuple[float, float, float, float, float]]:
        """Calculate precision and recall using sweep-based algorithm.

        Args:
            thresholds: Optional list of thresholds to override instance thresholds

        Returns:
            List of (threshold, precision, recall, precision_ci, recall_ci) tuples
        """
        if thresholds is None:
            thresholds = self.thresholds

        if not thresholds:
            thresholds = [1.0]  # Default to 100% threshold

        return self._calculate_optimised_pr_curve(thresholds)

    def _calculate_optimised_pr_curve(
        self, thresholds: list[float]
    ) -> list[tuple[float, float, float, float, float]]:
        """Optimised PR calculation using Union-Find sweep algorithm."""
        # Handle empty judgements case
        if len(self.judgements) == 0:
            # Return zeros for all thresholds when no judgements exist
            return [(t, 0.0, 0.0, 0.0, 0.0) for t in thresholds]

        # Initialise incremental state with cached validation data
        state = IncrementalState(
            self._validation_pairs, self._validation_counts, self._validation_leaves
        )

        results = []
        # Process thresholds in descending order (monotonic progression)
        for threshold in sorted(thresholds, reverse=True):
            # Incrementally add edges at this threshold
            state.add_threshold_edges(threshold, self.probabilities, self.root_leaf)

            # Calculate PR metrics from current state
            precision, recall, precision_ci, recall_ci = (
                state.calculate_current_metrics()
            )
            results.append((threshold, precision, recall, precision_ci, recall_ci))

        # Sort results by threshold ascending for consistent output
        return sorted(results, key=lambda x: x[0])


def compare_models(resolutions: list[ModelResolutionPath]) -> ModelComparison:
    """Compare metrics of models based on evaluation data.

    Args:
        resolutions: List of names of model resolutions to be compared.

    Returns:
        A model comparison object, listing metrics for each model.
    """
    return _handler.compare_models(resolutions)
