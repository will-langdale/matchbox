"""Status bar widget for entity resolution evaluation."""

from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

from matchbox.client.cli.eval.widgets.styling import get_display_text


class StatusBar(Widget):
    """Status bar widget with reactive attributes."""

    queue_position: reactive[int] = reactive(0)
    queue_total: reactive[int] = reactive(0)
    group_counts: reactive[dict[str, int]] = reactive({})
    current_group: reactive[str] = reactive("")
    status_message: reactive[str] = reactive("○ Ready")
    status_colour: reactive[str] = reactive("dim")

    def render(self) -> Table:
        """Render single-row status table."""
        table = Table.grid(expand=True)
        table.add_column(ratio=3)  # Left side
        table.add_column(ratio=1)  # Right side

        left = self._render_left()
        right = self._render_right()
        table.add_row(left, right)
        return table

    def _render_left(self) -> Text:
        """Render left side with entity progress and groups."""
        text = Text()

        if self.queue_total > 0:
            text.append(
                f"Entity {self.queue_position}/{self.queue_total}",
                style="bright_white",
            )
        else:
            text.append("No samples to evaluate", style="yellow")
            return text

        text.append(" | ", style="dim")

        if not self.group_counts:
            text.append("No groups assigned", style="dim")
        else:
            text.append("Groups: ", style="bright_white")
            for i, (group, count) in enumerate(self.group_counts.items()):
                if i > 0:
                    text.append("  ")

                display_text, colour = get_display_text(group, count)

                if group == self.current_group:
                    text.append(display_text, style=f"bold {colour} underline")
                else:
                    text.append(display_text, style=f"bold {colour}")
        text.append(" | ", style="dim")
        text.append("Press letter to select group", style="yellow")

        return text

    def _render_right(self) -> Text:
        """Render right side with status indicator."""
        text = Text()

        if self.status_message:
            text.append(self.status_message, style=self.status_colour)
        else:
            text.append("○ Ready", style="dim")

        return text
