"""Modal screens for entity resolution evaluation."""

from textwrap import dedent

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class HelpModal(ModalScreen):
    """Help screen showing commands and shortcuts."""

    def compose(self) -> ComposeResult:
        """Compose the help modal UI."""
        with Container(id="help-dialog"):
            yield Static("Entity resolution tool - Help", id="help-title")
            yield Static(
                dedent("""
                    Simple keyboard shortcuts (comparison view):
                    • Press A - Selects group A
                    • Press 1,3,5 - Assigns columns 1,3,5 to group A
                    • Press B - Switches to group B
                    • Press 2,4 - Assigns columns 2,4 to group B
                    • Press A - Switches back to group A
                    • Press 6 - Assigns column 6 to group A

                    Group selection:
                    • Any letter (a-z) selects that group (26 groups available)
                    • Only one group active at a time
                    • Clear visual feedback shows which group is selected

                    Column assignment:
                    • 1-9: Assign to columns 1-9
                    • 0: Assign to column 10 (if exists)
                    • Numbers only work when a group is selected

                    Navigation shortcuts:
                    • → or Enter - Next entity
                    • ← - Previous entity
                    • Space - Submit current judgement & fetch more samples
                    • Ctrl+G - Jump to entity number
                    • ? or F1 - Show this help
                    • Esc - Clear current group selection
                    • Ctrl+C or Ctrl+Q - Quit

                    Visual feedback:
                    • Records are columns, fields are rows
                    • Same field types grouped together for easy comparison
                    • Each group gets unique colour + symbol combination
                    • Column headers show group assignment with coloured symbols
                    • Status bar shows group counts with visual indicators
                    • Empty rows are automatically filtered out

                    Tips for speed:
                    • Press letter with left hand, numbers with right
                    • Same group always gets same colour/symbol for consistency
                    • Work in patterns: group obvious matches first
                    • Use Esc to clear if you get confused
                """).strip(),
                id="help-content",
            )
            yield Button("Close (Esc)", id="close-help")

    @on(Button.Pressed, "#close-help")
    def close_help(self) -> None:
        """Close the help modal."""
        self.dismiss()

    def on_key(self, event: events.Key) -> None:
        """Handle key events for closing the help modal."""
        if event.key == "escape" or event.key == "question_mark":
            self.dismiss()
