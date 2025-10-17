"""Comparison display table for entity resolution evaluation."""

from typing import Any

import polars as pl
from rich.table import Table
from textual.widget import Widget

from matchbox.client.cli.eval.state import EvaluationState
from matchbox.client.cli.eval.widgets.styling import GroupStyler
from matchbox.client.eval import EvaluationItem


class ComparisonDisplayTable(Widget):
    """Minimal table for side-by-side record comparison with colour highlighting."""

    def __init__(self, state: EvaluationState, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialise the comparison display table."""
        super().__init__(**kwargs)
        self.state = state
        self.state.add_listener(self._on_state_change)

    def render(self) -> Table:
        """Render the table with current state using Rich Table."""
        current = self.state.queue.current
        if not current or not hasattr(current, "display_dataframe"):
            return self._create_loading_table()

        return self._render_compact_view(current)

    def _create_loading_table(self) -> Table:
        """Create a simple loading table."""
        loading_table = Table(show_header=False, show_lines=False)
        loading_table.add_column("")
        loading_table.add_row("Loading...")
        return loading_table

    def _add_table_columns(self, table: Table, current: EvaluationItem) -> None:
        """Add styled columns to the table for each display column."""
        current_assignments = self.state.current_assignments
        for display_col_index, _ in enumerate(current.display_columns):
            col_num = display_col_index + 1
            duplicate_count = len(current.duplicate_groups[display_col_index])

            # Create header with duplicate count indicator
            header_text = (
                f"{col_num} (×{duplicate_count})"
                if duplicate_count > 1
                else str(col_num)
            )

            if display_col_index in current_assignments:
                group = current_assignments[display_col_index]
                colour, symbol = GroupStyler.get_style(group)
                header = f"[{colour}]{symbol} {header_text}[/]"
                table.add_column(header, style=colour, min_width=15, max_width=50)
            else:
                table.add_column(header_text, style="dim", min_width=15, max_width=50)

    def _render_compact_view(self, current: EvaluationItem) -> Table:
        """Render compact view - one row per field, deduplicated columns."""
        if current.display_dataframe.is_empty():
            return self._create_loading_table()

        table = Table(
            show_header=True,
            header_style="bold",
            show_lines=False,
            row_styles=[],
            box=None,
            padding=(0, 1),
        )
        table.add_column("Field", style="bright_white", min_width=20, max_width=50)
        self._add_table_columns(table, current)

        # Group by field_name and create compact rows
        field_groups = (
            current.display_dataframe.group_by("field_name")
            .agg([pl.col("leaf_id"), pl.col("value")])
            .sort("field_name")
        )

        for field_row in field_groups.iter_rows(named=True):
            field_name = field_row["field_name"]
            leaf_ids_in_group = field_row["leaf_id"]
            values_in_group = field_row["value"]

            leaf_to_value = dict(zip(leaf_ids_in_group, values_in_group, strict=True))

            row_data = [field_name]
            for representative_leaf_id in current.display_columns:
                row_data.append(leaf_to_value.get(representative_leaf_id, ""))

            if any(cell for cell in row_data[1:]):
                table.add_row(*row_data)

        return table

    def _on_state_change(self) -> None:
        """Handle state changes by refreshing the display."""
        self.refresh()
