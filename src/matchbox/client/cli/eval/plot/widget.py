"""Simplified plot widget for displaying precision-recall curves."""

import logging
import traceback

import numpy as np
from textual_plotext import PlotextPlot

from matchbox.client.cli.eval.plot.core import compute_pr_envelope, interpolate_pr_curve

logger = logging.getLogger(__name__)


class PRCurveDisplay(PlotextPlot):
    """Simplified widget for displaying precision-recall curves.

    This widget is now "dumb" - it just renders the plot based on state.
    All error handling and data validation is done at the app level.
    """

    def __init__(self, state, **kwargs):
        """Initialise the PR curve display widget."""
        super().__init__(**kwargs)
        self.state = state
        self.state.add_listener(self._on_state_change)
        self._has_plotted = False

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        self.update_plot()

    def update_plot(self) -> None:
        """Update the plot based on current state."""
        # Clear any existing data first
        self.plt.clear_data()
        self.plt.clear_figure()

        # Handle early exit conditions with simple checks
        if not self._should_plot():
            return

        # Generate plot or show error message
        try:
            self._generate_pr_plot()
        except Exception as e:  # noqa: BLE001
            self._show_error_message(e)

    def _should_plot(self) -> bool:
        """Simple check if we should attempt plotting."""
        if self.state.is_loading_eval_data:
            self.plt.title("Loading evaluation data...")
            self._has_plotted = False
            return False

        if self.state.eval_data_error:
            self.plt.title("Error loading data - check status")
            self._has_plotted = False
            return False

        if self.state.eval_data is None:
            self.plt.title("No evaluation data loaded")
            self._has_plotted = False
            return False

        return True

    def _generate_pr_plot(self) -> None:
        """Generate the precision-recall plot with confidence intervals."""
        # Get raw data from EvalData
        pr_data = self.state.eval_data.precision_recall()

        # Compute smooth envelope using PCHIP interpolation
        r_grid, p_upper, p_lower = compute_pr_envelope(pr_data)

        # Compute interpolated PR curve with extrapolation tracking
        r_curve, p_curve, is_extrapolated = interpolate_pr_curve(pr_data)

        # Check if we have plottable data (arrays must not be empty)
        if len(r_curve) == 0 or len(r_grid) == 0:
            self.plt.title("📊 Need more labeled samples to plot PR curve")
            self._has_plotted = False
            return

        # Plot confidence bounds in magenta
        self.plt.plot(r_grid, p_upper, color="magenta", marker="braille")
        self.plt.plot(r_grid, p_lower, color="magenta", marker="braille")

        # Plot PR curve - split by extrapolation status
        # Find transition points between interpolated and extrapolated regions
        transitions = np.where(np.diff(is_extrapolated.astype(int)))[0]
        indices = np.concatenate([[0], transitions + 1, [len(r_curve)]])

        # Plot each segment with appropriate marker style
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            segment_r = r_curve[start_idx:end_idx]
            segment_p = p_curve[start_idx:end_idx]

            if is_extrapolated[start_idx]:
                # Extrapolated region: use braille markers
                self.plt.plot(segment_r, segment_p, color="green", marker="braille")
            else:
                # Interpolated region: use fhd markers for better quality
                self.plt.plot(segment_r, segment_p, color="green", marker="fhd")

        self.plt.xlim(0, 1)
        self.plt.ylim(0, 1)
        self.plt.xlabel("Recall")
        self.plt.ylabel("Precision")
        self.plt.title("Precision-Recall Curve (PCHIP Envelope)")
        self._has_plotted = True

    def _show_error_message(self, error: Exception) -> None:
        """Show appropriate error message for plot failures."""
        error_str = str(error).lower()

        if "cannot be empty" in error_str and "judgement" in error_str:
            # Expected case when no judgements exist yet
            self.plt.title("📊 Submit some judgements first")
        elif (
            "at least 2 elements" in error_str or "at least 2 data points" in error_str
        ):
            # Expected case when insufficient unique data points exist
            # This can happen with very few judgements that all produce the same outcome
            self.plt.title("📊 Need more labeled samples to plot PR curve")
        elif isinstance(error, IndexError) and (
            "out of bounds" in error_str or "index" in error_str
        ):
            # Expected when arrays are empty (should now be prevented by early check)
            self.plt.title("📊 Need more labeled samples to plot PR curve")
        else:
            # Unexpected error - log details and show generic message
            logger.error(f"Plot generation failed - {type(error).__name__}: {error}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.plt.title("Plot generation failed")

        self._has_plotted = False

    def _on_state_change(self) -> None:
        """Handle state changes by updating the plot."""
        # Only update if widget is actually mounted AND plot is being displayed
        # This prevents wasteful background calculation when modal is dismissed
        if not self.is_mounted or not self.state.show_plot:
            return
        self.update_plot()
