"""Comparison display table for entity resolution evaluation."""

from typing import Any

from rich import box
from rich.table import Table
from textual.widget import Widget

from matchbox.client.cli.eval.widgets.styling import get_group_style
from matchbox.client.eval import EvaluationItem


class ComparisonDisplayTable(Widget):
    """Table for side-by-side record comparison with colour highlighting."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialise the comparison display table."""
        super().__init__(**kwargs)
        self.current_item: EvaluationItem | None = None
        self.col_start: int = 0
        self.col_end: int = 0
        self.row_start: int = 0
        self.row_end: int | None = None

    def load_comparison(
        self,
        item: EvaluationItem,
        col_start: int,
        col_end: int,
        row_start: int,
        row_end: int,
    ) -> None:
        """Load comparison data for a specific range of columns and rows.

        Args:
            item: The evaluation item to display
            col_start: Starting column index (inclusive)
            col_end: Ending column index (exclusive)
            row_start: Starting row index (inclusive)
            row_end: Ending row index (exclusive)
        """
        self.current_item = item
        self.col_start = col_start
        self.col_end = col_end
        self.row_start = row_start
        self.row_end = row_end
        self.refresh()

    def render(self) -> Table:
        """Render the table with current item and range."""
        if not self.current_item:
            return self._create_loading_table()

        table = Table(
            show_header=True,
            header_style="bold bright_white",
            show_lines=False,
            row_styles=["", "on #0a1628"],
            box=box.SIMPLE_HEAD,
            padding=(0, 1),
            expand=True,
            border_style="#E4A242",
        )

        self._add_table_columns(table, self.current_item)

        # Get all field names in order
        field_names = list(self.current_item.display_data.keys())

        # Determine which rows to show
        end_row = self.row_end if self.row_end is not None else len(field_names)
        rows_to_show = field_names[self.row_start : end_row]

        # Add rows
        for field_name in rows_to_show:
            values = self.current_item.display_data[field_name]
            # Extract only the values for columns in our range
            row_values = values[self.col_start : self.col_end]
            table.add_row(field_name, *row_values)

        return table

    def _create_loading_table(self) -> Table:
        """Create a simple loading table."""
        loading_table = Table(show_header=False, show_lines=False)
        loading_table.add_column("")
        loading_table.add_row("Loading...")
        return loading_table

    def _add_table_columns(self, table: Table, current: EvaluationItem) -> None:
        """Add styled columns to the table for the specified column range."""
        # First column for field names with distinct styling
        table.add_column(
            "",  # No header
            style="bright_white",
            min_width=20,
            max_width=50,
        )

        # Data columns - only for the specified range
        # Column numbers are relative to each tab (1-N per tab)
        for idx, display_col_index in enumerate(range(self.col_start, self.col_end)):
            # Column number within this tab (1-based, or 0 for 10th column)
            col_num = idx + 1
            if col_num == 10:
                col_num = 0  # 10th column shows as 0 (typed with "0" key)

            duplicate_count = len(current.duplicate_groups[display_col_index])

            header_text = (
                f"{col_num} (×{duplicate_count})"
                if duplicate_count > 1
                else str(col_num)
            )

            if display_col_index in current.assignments:
                group = current.assignments[display_col_index]
                symbol, colour = get_group_style(group)
                header = f"[{colour} bold]{symbol} {header_text}[/]"
                table.add_column(
                    header,
                    style=colour,
                    min_width=15,
                    max_width=50,
                )
            else:
                table.add_column(
                    f"[dim]{header_text}[/]",
                    style="dim",
                    min_width=15,
                    max_width=50,
                )
