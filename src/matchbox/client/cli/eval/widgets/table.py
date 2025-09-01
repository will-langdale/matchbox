"""Comparison display table for entity resolution evaluation."""

import polars as pl
from rich.table import Table
from textual.widget import Widget

from matchbox.client.cli.eval.state import EvaluationState
from matchbox.client.cli.eval.widgets.styling import GroupStyler


class ComparisonDisplayTable(Widget):
    """Minimal table for side-by-side record comparison with colour highlighting."""

    def __init__(self, state: EvaluationState, **kwargs):
        """Initialise the comparison display table."""
        super().__init__(**kwargs)
        self.state = state
        self.state.add_listener(self._on_state_change)

    def render(self) -> Table:
        """Render the table with current state using Rich Table."""
        current = self.state.queue.current
        if not current or not hasattr(current, "display_dataframe"):
            return self._create_loading_table()

        if self.state.compact_view_mode:
            return self._render_compact_view(current.display_dataframe)
        else:
            return self._render_detailed_view()

    def _create_loading_table(self) -> Table:
        """Create a simple loading table."""
        loading_table = Table(show_header=False, show_lines=False)
        loading_table.add_column("")
        loading_table.add_row("Loading...")
        return loading_table

    def _render_compact_view(self, display_df: pl.DataFrame) -> Table:
        """Render compact view - one row per field, deduplicated columns."""
        current = self.state.queue.current
        if not current or display_df.is_empty():
            return self._create_loading_table()

        # Create Rich Table
        table = Table(
            show_header=True,
            header_style="bold",
            show_lines=False,
            row_styles=[],
            box=None,
            padding=(0, 1),
        )

        # Add field name column
        table.add_column("Field", style="bright_white", min_width=20, max_width=50)

        # Add columns for each display column (deduplicated)
        current_assignments = self.state.current_assignments
        for display_col_index in range(len(current.display_columns)):
            col_num = display_col_index + 1
            duplicate_count = len(current.duplicate_groups[display_col_index])

            # Create header with duplicate count indicator
            if duplicate_count > 1:
                header_text = f"{col_num} (×{duplicate_count})"
            else:
                header_text = str(col_num)

            if display_col_index in current_assignments:
                group = current_assignments[display_col_index]
                colour, symbol = GroupStyler.get_style(group)
                header = f"[{colour}]{symbol} {header_text}[/]"
                table.add_column(header, style=colour, min_width=15, max_width=50)
            else:
                table.add_column(header_text, style="dim", min_width=15, max_width=50)

        # Group by field_name and create compact rows
        field_groups = (
            display_df.group_by("field_name")
            .agg([pl.col("leaf_id"), pl.col("value")])
            .sort("field_name")
        )  # Sort to maintain consistent field order

        # Filter out empty groups and create table rows
        for field_row in field_groups.iter_rows(named=True):
            field_name = field_row["field_name"]
            leaf_ids_in_group = field_row["leaf_id"]
            values_in_group = field_row["value"]

            # Create a mapping from leaf_id to value for this field
            leaf_to_value = {}
            for leaf_id, value in zip(leaf_ids_in_group, values_in_group, strict=True):
                leaf_to_value[leaf_id] = value

            # Build row data using representative leaf IDs
            row_data = [field_name]
            for representative_leaf_id in current.display_columns:
                cell_value = leaf_to_value.get(representative_leaf_id, "")
                row_data.append(cell_value)

            # Only add row if it has some data
            if any(cell for cell in row_data[1:]):  # Skip field name in check
                table.add_row(*row_data)

        return table

    def _render_detailed_view(self) -> Table:
        """Render detailed view - source attribution with deduplication."""
        current = self.state.queue.current
        if not current or current.display_dataframe.is_empty():
            return self._create_loading_table()

        # Create Rich Table with minimal styling - focus on colour
        table = Table(
            show_header=True,
            header_style="bold",
            show_lines=False,
            row_styles=[],
            box=None,
            padding=(0, 1),
        )

        # Add field name column
        table.add_column("Field", style="bright_white", min_width=20, max_width=50)

        # Add columns for each display column with group styling
        current_assignments = self.state.current_assignments
        for display_col_index, _ in enumerate(current.display_columns):
            col_num = display_col_index + 1
            duplicate_count = len(current.duplicate_groups[display_col_index])

            # Create header with duplicate count indicator
            if duplicate_count > 1:
                header_text = f"{col_num} (×{duplicate_count})"
            else:
                header_text = str(col_num)

            if display_col_index in current_assignments:
                group = current_assignments[display_col_index]
                colour, symbol = GroupStyler.get_style(group)
                header = f"[{colour}]{symbol} {header_text}[/]"
                table.add_column(header, style=colour, min_width=15, max_width=50)
            else:
                table.add_column(header_text, style="dim", min_width=15, max_width=50)

        # Group by field and source for detailed view
        field_source_groups = (
            current.display_dataframe.group_by(["field_name", "source_name"])
            .agg([pl.col("leaf_id"), pl.col("value")])
            .sort(["field_name", "source_name"])  # Sort for consistency
        )

        # Organize data for detailed view with source attribution
        field_source_data = {}
        for row in field_source_groups.iter_rows(named=True):
            field_name = row["field_name"]
            source_name = row["source_name"]
            leaf_ids = row["leaf_id"]
            values = row["value"]

            if field_name not in field_source_data:
                field_source_data[field_name] = {}

            # Map leaf_ids to values for this field+source combination
            for leaf_id, value in zip(leaf_ids, values, strict=True):
                field_source_data[field_name][f"{field_name} ({source_name})"] = {
                    leaf_id: value
                }

        # Add data rows with source attribution
        for field_name in sorted(field_source_data.keys()):
            source_data = field_source_data[field_name]
            # Add separator between field groups
            if len([r for r in table.rows]) > 0:
                separator_row = ["─" * 15] + ["─" * 8] * len(current.display_columns)
                table.add_row(*separator_row, style="dim")

            # Add rows for each source's version of this field
            for source_field_name in sorted(source_data.keys()):
                leaf_value_map = source_data[source_field_name]
                row_data = [source_field_name]

                # Build row using representative leaf IDs
                for representative_leaf_id in current.display_columns:
                    cell_value = leaf_value_map.get(representative_leaf_id, "")
                    row_data.append(cell_value)

                # Only add row if it has some data (empty row filtering)
                if any(cell.strip() for cell in row_data[1:]):
                    table.add_row(*row_data)

        return table

    def _on_state_change(self) -> None:
        """Handle state changes by refreshing the display."""
        self.refresh()
