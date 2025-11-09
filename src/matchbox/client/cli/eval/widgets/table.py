"""Comparison display table for entity resolution evaluation."""

from typing import Any

import polars as pl
from rich.text import Text
from textual import events
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import DataTable

from matchbox.client.cli.eval.widgets.styling import get_group_style
from matchbox.client.eval import EvaluationItem


class ComparisonDisplayTable(DataTable):
    """DataTable for comparing records with keyboard-driven assignment.

    Handles:
    - Displaying all unique records from a cluster (scrollable)
    - Keyboard shortcuts for assignment (a-z to select group, 1-0 to assign)
    - Reactive updates when data/assignments change
    - Message posting when assignments/current group change
    """

    current_item: reactive[EvaluationItem | None] = reactive(None)
    current_assignments: reactive[dict[int, str]] = reactive({}, init=False)
    current_group: reactive[str] = reactive("")

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the comparison display table."""
        super().__init__(**kwargs)
        self.zebra_stripes = True
        self.cursor_type = "none"
        self.show_cursor = False

    def watch_current_item(self, item: EvaluationItem | None) -> None:
        """Rebuild table when item changes (Textual reactive pattern)."""
        if not item:
            self.clear(columns=True)
            return

        unique_groups = item.get_unique_record_groups()

        # Build table with all unique records
        self.clear(columns=True)
        self.add_column("Field", key="field")

        # Add column for each unique group
        for i, group in enumerate(unique_groups):
            # Build header with position (1-indexed) and no assignment
            header = self._build_header(i + 1, group, None)
            self.add_column(header, key=f"col_{i}")

        # Add row for each field
        for field in item.fields:
            row_values = [Text(field.display_name)]
            for group in unique_groups:
                record_idx = group[0]  # Representative record from group
                row_data = item.records.filter(pl.col("leaf") == record_idx).row(
                    0, named=True
                )
                # Take first non-empty value from source columns
                value = next(
                    (
                        str(row_data.get(col, "")).strip()
                        for col in field.source_columns
                        if row_data.get(col)
                    ),
                    "",
                )
                row_values.append(Text(value))
            self.add_row(*row_values, key=field.display_name)

    def watch_current_assignments(self, assignments: dict[int, str]) -> None:
        """Rebuild column headers when assignments change (Textual reactive pattern)."""
        self._apply_assignment_styling()

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard shortcuts for assignment."""
        if event.key in "abcdefghijklmnopqrstuvwxyz":
            self.current_group = event.key
            # Notify app for status bar underline
            self.post_message(self.CurrentGroupChanged(event.key))
            event.stop()  # Consume event - don't pass to app

        elif event.key in "1234567890":
            key_num = int(event.key) if event.key != "0" else 10
            self._assign_visible_column(key_num - 1)  # Convert to 0-indexed
            event.stop()  # Consume event

    def _get_column_positions(self) -> list[int]:
        """Get visual positions for each data column.

        Returns:
            List of visual positions (1-indexed) for each data column
        """
        if not self.current_item:
            return []

        unique_groups = self.current_item.get_unique_record_groups()
        return list(range(1, len(unique_groups) + 1))

    def _apply_assignment_styling(self) -> None:
        """Apply assignment styling to all data columns.

        Rebuilds all data columns (headers and cells) from scratch, applying:
        - Current visual positions (1-indexed numbers)
        - Duplicate counts (×N if multiple records)
        - Assignment symbols and colours (if assigned)
        """
        if not self.current_item:
            return

        unique_groups = self.current_item.get_unique_record_groups()
        positions = self._get_column_positions()

        # Save all data columns (skip column 0 which is "Field")
        columns_to_rebuild = []
        for idx in range(1, len(self.ordered_columns)):
            col = self.ordered_columns[idx]
            col_key = col.key
            group_idx = idx - 1  # Convert column index to group index

            # Extract all cell data for this column
            col_data = [
                (row_key, self.get_cell(row_key, col_key)) for row_key in self.rows
            ]

            # Get assignment for this group index
            assignment = self.current_assignments.get(group_idx)

            columns_to_rebuild.append((col_key, col_data, group_idx, assignment))

        # Remove all data columns
        for col_key, _, _, _ in columns_to_rebuild:
            self.remove_column(col_key)

        # Re-add columns with new headers
        for col_key, col_data, group_idx, assignment in columns_to_rebuild:
            group = unique_groups[group_idx]
            visual_pos = positions[group_idx]

            # Build header with current position and assignment
            header = self._build_header(visual_pos, group, assignment)

            # Add column with new header
            self.add_column(header, key=col_key)

            # Restore cell data with colour if assigned
            if assignment:
                _, colour = get_group_style(assignment)
                for row_key, text_value in col_data:
                    coloured_text = Text()
                    coloured_text.append_text(text_value)
                    coloured_text.stylize(colour)
                    self.update_cell(row_key, col_key, coloured_text)
            else:
                # Restore cells without colour
                for row_key, value in col_data:
                    self.update_cell(row_key, col_key, value)

    def _assign_visible_column(self, visible_idx: int) -> None:
        """Assign Nth visible column (0-indexed) to current group.

        Args:
            visible_idx: Index in the list of visible columns (0-9 for keys 1-0)
        """
        if not self.current_item or not self.current_group:
            return

        unique_groups = self.current_item.get_unique_record_groups()

        if visible_idx >= len(unique_groups):
            return  # Key out of range

        # Post message to app to update assignments
        self.post_message(self.AssignmentMade(visible_idx, self.current_group))

    def _build_header(
        self, visual_pos: int, group: list[int], assignment: str | None = None
    ) -> Text:
        """Build header with position, count, and optional assignment.

        Args:
            visual_pos: Visual position to display (1-indexed)
            group: List of record indices in this group (for duplicate count)
            assignment: Optional assignment group letter (e.g., 'a', 'b')

        Returns:
            Rich Text object with complete styling including symbols and colours
        """
        # Show duplicate count if > 1
        dup_count = len(group)
        count_text = f" (×{dup_count})" if dup_count > 1 else ""

        header = Text()

        if assignment:
            # Get symbol and colour for this assignment
            symbol, colour = get_group_style(assignment)
            header.append(f"{symbol} {visual_pos}{count_text}")
            header.stylize(colour)
        else:
            # Unassigned column - plain dim style
            header.append(f"{visual_pos}{count_text}", style="dim")

        return header

    class AssignmentMade(Message):
        """Posted when a single column assignment is made."""

        def __init__(self, column_idx: int, group: str) -> None:
            """Initialise assignment made message."""
            super().__init__()
            self.column_idx = column_idx
            self.group = group

    class CurrentGroupChanged(Message):
        """Posted when current group selection changes."""

        def __init__(self, group: str) -> None:
            """Initialise current group changed message."""
            super().__init__()
            self.group = group
