"""3D Precision-Recall Confidence Envelopes.

Core insight: PR curves naturally exist in 3D space (threshold, precision, recall).
Confidence intervals are cross-sections of a 3D "tube" around the true curve.
This module projects that 3D tube to clean 2D confidence bounds.

Key innovation: Uses PCHIP interpolation to preserve monotonic PR curve behaviour
while applying probabilistic uncertainty scaling based on data coverage.

The story:
1. Problem: PR confidence intervals create messy overlapping rectangles in 2D
2. Insight: Work in natural 3D parameter space (threshold axis) instead
3. Solution: PCHIP for monotonic curves + probabilistic uncertainty scaling
4. Result: Clean 2D bounds that respect PR curve physics

Example usage:
    data = [(threshold, precision, recall, p_ci, r_ci), ...]
    recall_grid, upper_bounds, lower_bounds = compute_pr_envelope(data)
    plotext_pr_envelope(data)  # plot in terminal
"""

import logging
from collections.abc import Callable

import numpy as np
import plotext as pltxt
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)

# Configuration constants
COVERAGE_SIGMA = 0.25  # Exponential decay rate for coverage probability
MAX_UNCERTAINTY = 5.0  # Maximum uncertainty multiplier in uncovered regions
RECALL_TOLERANCE = 0.02  # Tolerance for threshold-recall matching
THRESHOLD_SEARCH_POINTS = 1000  # Resolution for threshold search grid
MIN_COVERAGE_PROB = 0.01  # Minimum coverage probability (avoids division by zero)


def fit_tube(
    pr_data: list[tuple[float, float, float, float, float]],
) -> tuple[Callable, Callable, Callable, Callable]:
    """Fit 3D parametric tube using PCHIP (Piecewise Cubic Hermite Interpolating).

    PCHIP is essential for PR curves because:
    - Monotonicity preservation: Maintains decreasing precision with increasing recall
    - Shape-preserving: Prevents oscillations standard splines create with noisy data
    - Local behaviour: Uses only nearby points, avoiding distant outlier influence

    This ensures PR curves behave physically without artificial wiggles that violate
    the fundamental precision-recall constraint.
    """
    if len(pr_data) < 2:
        raise ValueError(
            "Need at least 2 data points to compute meaningful PR envelope. "
            "Single points cannot define a curve or confidence bounds."
        )

    data = np.array(pr_data)
    thresholds, precisions, recalls, p_widths, r_widths = data.T

    # Sort by threshold for proper interpolation
    idx = np.argsort(thresholds)
    t, p, r, pw, rw = (
        thresholds[idx],
        precisions[idx],
        recalls[idx],
        p_widths[idx],
        r_widths[idx],
    )

    # PCHIP interpolation for monotonicity preservation
    return (
        PchipInterpolator(t, p, extrapolate=True),
        PchipInterpolator(t, r, extrapolate=True),
        PchipInterpolator(t, pw, extrapolate=True),
        PchipInterpolator(t, rw, extrapolate=True),
    )


def compute_uncertainty(
    recall_values: np.ndarray,
    pr_data: list[tuple[float, float, float, float, float]],
    recall_func: Callable,
) -> np.ndarray:
    """Vectorised uncertainty scaling based on data coverage probability.

    Returns higher uncertainty multipliers where data coverage is sparse.
    Uses exponential decay: high probability near data, low probability far from data.
    """
    data_thresholds = np.array([d[0] for d in pr_data])
    data_recalls = recall_func(data_thresholds)

    # Vectorised distance computation to all data points
    distances = np.abs(recall_values[:, None] - data_recalls[None, :])
    min_distances = np.min(distances, axis=1)

    # Coverage probability with exponential decay
    coverage_probs = np.exp(-((min_distances / COVERAGE_SIGMA) ** 2))
    coverage_probs = np.clip(coverage_probs, MIN_COVERAGE_PROB, 1.0)

    # Uncertainty scaling: inverse relationship with coverage
    uncertainty_multipliers = 1.0 + (MAX_UNCERTAINTY - 1.0) * (
        1.0 - np.sqrt(coverage_probs)
    )

    return uncertainty_multipliers


def project_to_envelope(
    pr_data: list[tuple[float, float, float, float, float]],
    precision_func: Callable,
    recall_func: Callable,
    precision_width_func: Callable,
    recall_width_func: Callable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised projection of 3D tube to 2D envelope preserving multi-threshold.

    Key insight: Multiple thresholds can produce similar recall values. Must find ALL
    valid matches and compute envelope across all possibilities, not just closest match.
    """
    recall_grid = np.linspace(0, 1, 101)
    t_search = np.linspace(0, 1, THRESHOLD_SEARCH_POINTS)

    # Vectorised distance computation: (thresholds, recall_targets)
    recall_search_values = recall_func(t_search)
    recall_distances = np.abs(recall_search_values[:, None] - recall_grid[None, :])

    # Boolean mask of ALL valid threshold-recall matches
    valid_mask = recall_distances < RECALL_TOLERANCE  # Shape: (1000, 101)

    # Compute uncertainty scaling for all recall values
    uncertainty_multipliers = compute_uncertainty(recall_grid, pr_data, recall_func)

    # Vectorised precision and width computation for all thresholds
    precision_centers = precision_func(t_search)  # Shape: (1000,)
    precision_widths = precision_width_func(t_search)  # Shape: (1000,)

    # Broadcast to create bounds matrix: (thresholds, recall_targets)
    p_centers_2d = precision_centers[:, None]  # (1000, 1)
    p_widths_2d = (
        precision_widths[:, None] * uncertainty_multipliers[None, :]
    )  # (1000, 101)

    # Compute all possible bounds
    upper_bounds_matrix = p_centers_2d + p_widths_2d
    lower_bounds_matrix = p_centers_2d - p_widths_2d

    # Mask invalid threshold-recall combinations
    upper_bounds_matrix[~valid_mask] = -np.inf  # Will be ignored in max
    lower_bounds_matrix[~valid_mask] = np.inf  # Will be ignored in min

    # Take envelope across all valid thresholds for each recall
    p_upper = np.max(upper_bounds_matrix, axis=0)
    p_lower = np.min(lower_bounds_matrix, axis=0)

    # Handle recall values with no valid matches (pure extrapolation)
    no_matches = ~np.any(valid_mask, axis=0)
    if np.any(no_matches):
        # Use closest threshold with boosted uncertainty
        closest_indices = np.argmin(recall_distances, axis=0)[no_matches]
        p_centers_fallback = precision_centers[closest_indices]
        p_widths_fallback = (
            precision_widths[closest_indices]
            * uncertainty_multipliers[no_matches]
            * 2.0
        )

        p_upper[no_matches] = p_centers_fallback + p_widths_fallback
        p_lower[no_matches] = p_centers_fallback - p_widths_fallback

    # Enforce constraints
    p_upper = np.clip(p_upper, 0, 1)
    p_lower = np.clip(p_lower, 0, 1)
    p_lower = np.minimum(p_lower, p_upper)

    return recall_grid, p_upper, p_lower


def _ensure_origin_point(
    pr_data: list[tuple[float, float, float, float, float]],
) -> list[tuple[float, float, float, float, float]]:
    """Ensure PR data includes the origin point (recall=0, precision=1).

    For all precision-recall curves, the point (recall=0, precision=1) is
    mathematically valid and represents the most conservative threshold where
    no predictions are made. This ensures we always have at least one point,
    preventing interpolation failures with sparse data.

    Args:
        pr_data: List of (threshold, precision, recall, p_ci, r_ci) tuples

    Returns:
        PR data with origin point guaranteed to be present
    """
    if not pr_data:
        # No data at all - return just the origin point
        return [(1.0, 1.0, 0.0, 0.0, 0.0)]

    # Check if we already have a point at recall ≈ 0 (within small tolerance)
    has_origin = any(abs(recall) < 0.001 for _, _, recall, _, _ in pr_data)

    if has_origin:
        return pr_data

    # Prepend the origin point
    # threshold=1.0 (max threshold), precision=1.0 (perfect), recall=0.0 (none found)
    # p_ci=0.0, r_ci=0.0 (no uncertainty at boundary)
    origin = (1.0, 1.0, 0.0, 0.0, 0.0)
    return [origin] + pr_data


def _deduplicate_recall_values(
    pr_data: list[tuple[float, float, float, float, float]],
) -> list[tuple[float, float, float, float, float]]:
    """Remove duplicate recall values by averaging precision values.

    When multiple thresholds produce identical recall values, we average their
    precision values to create a single representative point. This preserves
    the essential precision-recall relationship while ensuring strictly
    increasing recall values required by PCHIP interpolation.

    Analytical choice: Simple arithmetic mean of precision values for duplicates.
    This approach treats all measurements equally and provides a balanced
    representation of precision at each recall level, without introducing
    bias toward any particular threshold.

    Args:
        pr_data: List of (threshold, precision, recall, p_ci, r_ci) tuples

    Returns:
        List of deduplicated tuples with unique recall values, sorted by recall
    """
    if not pr_data:
        return pr_data

    # Group by recall value
    recall_groups = {}
    for threshold, precision, recall, p_ci, r_ci in pr_data:
        if recall not in recall_groups:
            recall_groups[recall] = []
        recall_groups[recall].append((threshold, precision, p_ci, r_ci))

    # Process each recall group
    deduplicated = []
    for recall, group in recall_groups.items():
        if len(group) == 1:
            # Single point, keep as-is
            threshold, precision, p_ci, r_ci = group[0]
            deduplicated.append((threshold, precision, recall, p_ci, r_ci))
        else:
            # Multiple points with same recall - average precision values
            avg_precision = sum(precision for _, precision, _, _ in group) / len(group)
            avg_p_ci = sum(p_ci for _, _, p_ci, _ in group) / len(group)
            avg_r_ci = sum(r_ci for _, _, _, r_ci in group) / len(group)

            # Use threshold from first occurrence
            first_threshold = group[0][0]

            deduplicated.append(
                (
                    first_threshold,
                    avg_precision,
                    recall,
                    avg_p_ci,
                    avg_r_ci,
                )
            )

    # Sort by recall to ensure strictly increasing sequence
    return sorted(deduplicated, key=lambda x: x[2])


def interpolate_pr_curve(
    pr_data: list[tuple[float, float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a smooth PR curve through the actual data points.

    Uses PCHIP interpolation to maintain monotonicity while creating a smooth
    curve through the precision-recall data points. This gives us the central
    PR curve independent of the confidence envelope.

    Args:
        pr_data: List of (threshold, precision, recall, p_ci, r_ci) tuples

    Returns:
        recall_grid: Array of recall values from 0 to 1
        precision_curve: Interpolated precision values at recall_grid points
        is_extrapolated: Boolean array indicating extrapolated regions

    Mathematical foundation:
    - Sorts data by recall to ensure proper ordering
    - Uses PCHIP to preserve monotonic decreasing behaviour
    - Allows natural extrapolation beyond observed data
    - Tracks which regions are interpolated vs extrapolated
    """
    # Ensure we have the origin point (recall=0, precision=1)
    pr_data_with_origin = _ensure_origin_point(pr_data)

    # Deduplicate recall values to ensure strictly increasing sequence
    deduplicated_data = _deduplicate_recall_values(pr_data_with_origin)

    # Safety check: ensure we have enough points for interpolation
    if len(deduplicated_data) < 2:
        logger.warning(
            f"Insufficient unique data points after deduplication: "
            f"got {len(deduplicated_data)}, need at least 2. "
            f"Submit more judgements with different outcomes."
        )
        return np.array([]), np.array([]), np.array([])

    # Extract data (now guaranteed to have unique recall values)
    _, precisions, recalls, _, _ = zip(*deduplicated_data, strict=False)

    # Sort by recall for proper interpolation
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]

    # Create PCHIP interpolator with extrapolation enabled
    pr_curve_func = PchipInterpolator(
        sorted_recalls, sorted_precisions, extrapolate=True
    )

    # Evaluate at standard grid
    recall_grid = np.linspace(0, 1, 101)
    precision_curve = pr_curve_func(recall_grid)

    # Track which regions are extrapolated
    min_observed_recall = sorted_recalls[0]
    max_observed_recall = sorted_recalls[-1]
    is_extrapolated = (recall_grid < min_observed_recall) | (
        recall_grid > max_observed_recall
    )

    # Ensure valid precision values
    precision_curve = np.clip(precision_curve, 0, 1)

    return recall_grid, precision_curve, is_extrapolated


def compute_pr_envelope(
    pr_data: list[tuple[float, float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Main function: Probabilistic 3D approach using monotonic PCHIP interpolation.

    Pipeline:
    1. Fit 3D parametric tube using shape-preserving PCHIP interpolation
    2. Project tube to 2D envelope with vectorised probabilistic uncertainty scaling
    3. Return smooth, monotonic confidence bounds

    Theoretical foundation: Works with continuous mathematical objects throughout,
    discretising only for final visualisation.
    """
    # Ensure we have the origin point (recall=0, precision=1)
    pr_data_with_origin = _ensure_origin_point(pr_data)

    # Safety check: need at least 2 points for meaningful envelope calculation
    if len(pr_data_with_origin) < 2:
        logger.warning(
            f"Insufficient data points for PR envelope: "
            f"got {len(pr_data_with_origin)}, need at least 2. "
            f"Submit more judgements to compute confidence bounds."
        )
        return np.array([]), np.array([]), np.array([])

    # Fit monotonic 3D tube
    precision_func, recall_func, precision_width_func, recall_width_func = fit_tube(
        pr_data_with_origin
    )

    # Vectorised projection to 2D envelope
    return project_to_envelope(
        pr_data_with_origin,
        precision_func,
        recall_func,
        precision_width_func,
        recall_width_func,
    )


def plotext_pr_envelope(
    pr_data: list[tuple[float, float, float, float, float]],
    title: str = "3D PCHIP PR Envelope",
) -> None:
    """Plot PR envelope using plotext in the terminal.

    Terminal-based visualisation showing:
    - Green line: PR curve (fhd for interpolated, braille for extrapolated)
    - Magenta lines: Upper and lower confidence bounds

    No fill, just the essential curves for clean visualisation.

    Args:
        pr_data: List of (threshold, precision, recall, p_ci, r_ci) tuples
        title: Plot title for the terminal display
    """
    if not pr_data:
        logger.warning("No data to plot")
        return

    # Extract original data
    _, _, _, _, _ = zip(*pr_data, strict=False)

    # Compute envelope using the existing compute_pr_envelope function
    r_grid, p_upper, p_lower = compute_pr_envelope(pr_data)

    # Compute interpolated PR curve with extrapolation tracking
    r_curve, p_curve, is_extrapolated = interpolate_pr_curve(pr_data)

    # Clear any previous plot
    pltxt.clf()

    # Plot confidence bounds in red
    pltxt.plot(r_grid, p_upper, color="magenta+", marker="braille")
    pltxt.plot(r_grid, p_lower, color="magenta+", marker="braille")

    # Plot PR curve - split by extrapolation
    transitions = np.where(np.diff(is_extrapolated.astype(int)))[0]
    indices = np.concatenate([[0], transitions + 1, [len(r_curve)]])

    for i in range(len(indices) - 1):
        start_idx = indices[i]
        end_idx = indices[i + 1]
        segment_r = r_curve[start_idx:end_idx]
        segment_p = p_curve[start_idx:end_idx]

        if is_extrapolated[start_idx]:
            pltxt.plot(segment_r, segment_p, color="green", marker="braille")
        else:
            pltxt.plot(segment_r, segment_p, color="green", marker="fhd")

    # Configure plot
    pltxt.xlim(0, 1)
    pltxt.ylim(0, 1)
    pltxt.xlabel("Recall")
    pltxt.ylabel("Precision")
    pltxt.title(title)

    # Show the plot
    pltxt.show()
