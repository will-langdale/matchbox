"""Textual-based entity resolution evaluation tool."""

from pathlib import Path
from textwrap import dedent

import polars as pl
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Static,
)

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.cli.eval.utils import get_samples
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import MatchboxClientSettingsException
from matchbox.common.graph import DEFAULT_RESOLUTION, ModelResolutionName


class GroupStyler:
    """Generate consistent colors and symbols for any group name using hashing."""

    # High contrast colors that work well in terminals
    COLORS = [
        "red",
        "blue",
        "green",
        "yellow",
        "magenta",
        "cyan",
        "white",
        "bright_red",
        "bright_blue",
        "bright_green",
        "bright_yellow",
        "bright_magenta",
        "bright_cyan",
        "bright_white",
    ]

    # Distinct Unicode symbols for visual differentiation
    SYMBOLS = [
        "■",
        "●",
        "▲",
        "◆",
        "★",
        "⬢",
        "♦",
        "▼",
        "○",
        "△",
        "◇",
        "☆",
        "⬡",
        "✦",
        "✧",
        "⟐",
    ]

    @classmethod
    def get_style(cls, group_name: str) -> tuple[str, str]:
        """Get consistent color and symbol for a group name using hash."""
        # Use different hash seeds for color vs symbol to avoid correlation
        color_idx = hash(group_name + "_color_seed") % len(cls.COLORS)
        symbol_idx = hash(group_name + "_symbol_seed") % len(cls.SYMBOLS)

        color = cls.COLORS[color_idx]
        symbol = cls.SYMBOLS[symbol_idx]

        return color, symbol

    @classmethod
    def get_display_text(cls, group_name: str, count: int) -> tuple[str, str]:
        """Get formatted display text with color and symbol."""
        color, symbol = cls.get_style(group_name)
        text = f"{symbol} {group_name.upper()} ({count})"
        return text, color


class KeyboardShortcutHandler:
    """Handles keyboard shortcuts for fast group assignment."""

    def __init__(self):
        """Initialize the keyboard shortcut handler."""
        self.held_letters = []  # Track currently held letter keys
        self.number_keys = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "0": 10,
        }

    def on_key_down(self, key: str) -> None:
        """Handle key press down events."""
        if key.isalpha() and len(key) == 1:
            # Add letter to held combination if not already present
            letter = key.lower()
            if letter not in self.held_letters:
                self.held_letters.append(letter)

    def on_key_up(self, key: str) -> None:
        """Handle key release events."""
        if key.isalpha() and len(key) == 1:
            # Remove letter from held combination
            letter = key.lower()
            if letter in self.held_letters:
                self.held_letters.remove(letter)

    def get_current_group(self) -> str:
        """Get the current group name from held letters."""
        return "".join(self.held_letters) if self.held_letters else ""

    def parse_number_key(self, key: str) -> int | None:
        """Convert number key to row number."""
        return self.number_keys.get(key)


class StatusBar(Widget):
    """Widget to display group assignments with colored indicators."""

    groups = reactive({})

    def render(self) -> Text:
        """Render group status with colored symbols and dynamic colors."""
        if not self.groups:
            return Text("No groups assigned", style="dim")

        text = Text()

        for i, (group, count) in enumerate(self.groups.items()):
            if i > 0:
                text.append("  ")

            # Use GroupStyler for consistent color and symbol
            display_text, color = GroupStyler.get_display_text(group, count)
            text.append(display_text, style=f"bold {color}")

        return text

    def update_groups(self, groups: dict[str, int]) -> None:
        """Update group counts."""
        self.groups = groups


class RecordTable(DataTable):
    """Enhanced DataTable for displaying entity records with group coloring."""

    def __init__(self, **kwargs) -> None:
        """Initialize RecordTable with group assignment tracking."""
        super().__init__(**kwargs)
        self.group_assignments = {}  # row_index -> group_name
        self.cursor_type = "row"

    def set_data(self, df: pl.DataFrame) -> None:
        """Load data from Polars DataFrame."""
        self.clear(columns=True)

        # Add row number column
        columns = ["#"] + [col for col in df.columns if col != "leaf"]
        for col in columns:
            self.add_column(col, key=col)

        # Add data rows
        for i, row in enumerate(df.rows()):
            display_row = [str(i + 1)] + [
                str(val) if val is not None else "" for val in row[:-1]
            ]  # exclude leaf column
            self.add_row(*display_row, key=str(i))

    def assign_group(self, row_numbers: list[int], group: str) -> None:
        """Assign rows to a group and update styling."""
        for row_num in row_numbers:
            # Convert 1-based to 0-based indexing
            row_index = row_num - 1
            if 0 <= row_index < self.row_count:
                self.group_assignments[row_index] = group
                self._update_row_style(row_index, group)

    def _update_row_style(self, row_index: int, group: str) -> None:
        """Update the visual style of a row based on its group using dynamic styling."""
        # Use GroupStyler to get consistent color for any group name
        color, _symbol = GroupStyler.get_style(group)

        # Create style based on group length for visual distinction
        if len(group) > 1:
            # Extended groups (aa, bb, xyz, etc.) use bold styling with color
            style_name = f"bold {color}"
        else:
            # Single letter groups use background color
            style_name = f"on {color}"

        # Apply the style to the row
        try:
            row_key = str(row_index)
            if hasattr(self, "add_row_class"):
                # If Textual supports row classes (newer versions)
                self.add_row_class(row_key, f"group-{group}")
            else:
                # For current Textual version, store the style info
                if not hasattr(self, "_row_styles"):
                    self._row_styles = {}
                self._row_styles[row_index] = style_name
        except Exception:
            # Fallback - styling might not be fully supported
            pass

    def get_group_counts(self) -> dict[str, int]:
        """Get count of rows in each group."""
        counts = {}
        for group in self.group_assignments.values():
            counts[group] = counts.get(group, 0) + 1

        # Include unassigned count if there are unassigned rows
        assigned_count = sum(counts.values())
        remaining_count = self.row_count - assigned_count
        if remaining_count > 0:
            counts["unassigned"] = remaining_count

        return counts

    def get_judgement_groups(self, leaf_ids: list[int]) -> list[list[int]]:
        """Convert group assignments to judgement format."""
        groups = {}

        # Group leaf IDs by assignment
        for row_index, group in self.group_assignments.items():
            if group not in groups:
                groups[group] = []
            groups[group].append(leaf_ids[row_index])

        # Add unassigned rows to default group 'a' (for judgement format compatibility)
        assigned_rows = set(self.group_assignments.keys())
        unassigned_leaf_ids = [
            leaf_ids[i] for i in range(len(leaf_ids)) if i not in assigned_rows
        ]

        if unassigned_leaf_ids:
            if "a" not in groups:
                groups["a"] = []
            groups["a"].extend(unassigned_leaf_ids)

        # De-duplicate each group to avoid validation errors
        return [list(set(group)) for group in groups.values()]


class HelpModal(ModalScreen):
    """Help screen showing commands and shortcuts."""

    def compose(self) -> ComposeResult:
        """Compose the help modal UI."""
        with Container(id="help-dialog"):
            yield Static("Entity Resolution Tool - Help", id="help-title")
            yield Static(
                dedent("""
                    Fast Keyboard Shortcuts:
                    • Hold A, press 1,3,5 - Assign rows 1,3,5 to group A
                    • Hold Q, press 2 - Assign row 2 to group Q
                    • Hold A+S, press 4 - Assign row 4 to group AS
                    • Hold A+S+D, press 6,7 - Assign rows 6,7 to group ASD

                    Multi-letter Groups:
                    • Hold letters in sequence: A → A+S → A+S+D → A+S+D+F
                    • Release letters to build different combinations
                    • Any alphabetic combination works (QWERTY, HJKL, etc.)

                    Number Keys:
                    • 1-9: Assign to rows 1-9
                    • 0: Assign to row 10 (if exists)

                    Navigation Shortcuts:
                    • → or Enter - Next entity
                    • ← - Previous entity  
                    • Ctrl+G - Jump to entity number
                    • ? or F1 - Show this help
                    • Esc - Clear current assignments
                    • Ctrl+Q - Quit

                    Visual Feedback:
                    • Each group gets unique color + symbol combination
                    • Status bar shows group counts with visual indicators
                    • Help text shows currently held group

                    Tips for Speed:
                    • Use comfortable key combinations (ASDF, QWER, etc.)
                    • Hold letters with left hand, tap numbers with right
                    • Same group always gets same color/symbol for consistency
                    • Work in patterns: group obvious matches first
                """).strip(),
                id="help-content",
            )
            yield Button("Close (Esc)", id="close-help")

    @on(Button.Pressed, "#close-help")
    def close_help(self) -> None:
        """Close the help modal."""
        self.dismiss()

    def on_key(self, event) -> None:
        """Handle key events for closing the help modal."""
        if event.key == "escape" or event.key == "question_mark":
            self.dismiss()


class EntityResolutionApp(App):
    """Main Textual application for entity resolution evaluation."""

    CSS_PATH = Path(__file__).parent / "styles.css"

    BINDINGS = [
        ("right,enter", "next_entity", "Next"),
        ("left", "previous_entity", "Previous"),
        ("ctrl+g", "jump_to_entity", "Jump"),
        ("question_mark,f1", "show_help", "Help"),
        ("escape", "clear_assignments", "Clear"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        resolution: ModelResolutionName = DEFAULT_RESOLUTION,
        num_samples: int = 100,
        user: str | None = None,
        warehouse: str | None = None,
    ):
        """Initialize the Entity Resolution App."""
        super().__init__()
        self.resolution = resolution
        self.num_samples = num_samples
        self.user = user
        self.warehouse = warehouse
        self.current_entity = 0
        self.samples = {}
        self.user_id = None
        self.user_name = None
        self.current_df = None
        self.current_cluster_id = None
        self.session_file = Path("eval_session.json")
        self.keyboard_handler = KeyboardShortcutHandler()

    async def on_mount(self) -> None:
        """Initialize the application."""
        await self.authenticate()
        await self.load_samples()
        if self.samples:
            await self.load_entity(0)

    async def authenticate(self) -> None:
        """Authenticate with the server."""
        # Use injected user or fall back to settings
        user_name = self.user or settings.user
        if not user_name:
            raise MatchboxClientSettingsException("User name is unset.")

        self.user_name = user_name
        self.user_id = _handler.login(user_name=self.user_name)

    async def load_samples(self) -> None:
        """Load evaluation samples from the server."""
        # Build clients dict if warehouse URL is provided
        clients = {}

        if self.warehouse:
            # If warehouse URL is provided, create engine and disable default client
            from sqlalchemy import create_engine

            warehouse_engine = create_engine(self.warehouse)
            # We don't know the exact location name, so we'll use default client
            # but patch the settings temporarily
            original_warehouse = getattr(settings, "default_warehouse", None)
            settings.default_warehouse = self.warehouse
            try:
                self.samples = get_samples(
                    n=self.num_samples,
                    resolution=self.resolution,
                    user_id=self.user_id,
                    clients=clients,
                    use_default_client=True,
                )
            finally:
                # Clean up the engine
                warehouse_engine.dispose()
                # Restore original setting
                if original_warehouse is not None:
                    settings.default_warehouse = original_warehouse
                elif hasattr(settings, "default_warehouse"):
                    delattr(settings, "default_warehouse")
        else:
            # Use default behavior
            self.samples = get_samples(
                n=self.num_samples,
                resolution=self.resolution,
                user_id=self.user_id,
                use_default_client=True,
            )

    async def load_entity(self, entity_index: int) -> None:
        """Load a specific entity for evaluation."""
        if not self.samples:
            return

        cluster_ids = list(self.samples.keys())
        if 0 <= entity_index < len(cluster_ids):
            self.current_entity = entity_index
            self.current_cluster_id = cluster_ids[entity_index]
            self.current_df = self.samples[self.current_cluster_id]

            # Update UI
            table = self.query_one(RecordTable)
            table.set_data(self.current_df)
            table.group_assignments.clear()

            self.update_navigation_header()
            self.update_status_bar()

    def compose(self) -> ComposeResult:
        """Compose the main application UI."""
        yield Header()
        yield Static("Loading...", classes="navigation-header", id="nav-header")
        yield Container(
            StatusBar(classes="status-bar", id="status-bar"),
            RecordTable(id="record-table"),
            Static(
                "Hold letters + press numbers for fast assignment",
                classes="command-area",
                id="help-text",
            ),
            id="main-container",
        )
        yield Footer()

    def update_navigation_header(self) -> None:
        """Update the navigation header with current progress."""
        total = len(self.samples)
        current = self.current_entity + 1
        nav_text = f"Entity {current}/{total} | → Next | ← Previous | ? Help"

        header = self.query_one("#nav-header", Static)
        header.update(nav_text)

    def update_status_bar(self) -> None:
        """Update the status bar with current group assignments."""
        table = self.query_one(RecordTable)
        groups = table.get_group_counts()

        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_groups(groups)

    def on_key(self, event) -> None:
        """Handle keyboard events for group assignment shortcuts."""
        key = event.key

        # Handle letter key presses (build group combination)
        if key.isalpha() and len(key) == 1:
            self.keyboard_handler.on_key_down(key)
            self.update_help_text()
            return

        # Handle number key presses (assign rows to group)
        row_number = self.keyboard_handler.parse_number_key(key)
        if row_number is not None:
            current_group = self.keyboard_handler.get_current_group()
            if current_group:  # Only assign if we have a group selected
                table = self.query_one(RecordTable)
                if 1 <= row_number <= table.row_count:
                    table.assign_group([row_number], current_group)
                    self.update_status_bar()
            return

    def on_key_up(self, event) -> None:
        """Handle key release events."""
        key = event.key
        if key.isalpha() and len(key) == 1:
            self.keyboard_handler.on_key_up(key)
            self.update_help_text()

    def update_help_text(self) -> None:
        """Update the help text to show current group selection."""
        current_group = self.keyboard_handler.get_current_group()
        if current_group:
            _color, symbol = GroupStyler.get_style(current_group)
            help_text = (
                f"Group {symbol} {current_group.upper()} selected - "
                "press numbers to assign rows"
            )
        else:
            help_text = "Hold letters + press numbers for fast assignment"

        help_widget = self.query_one("#help-text", Static)
        help_widget.update(help_text)

    async def action_next_entity(self) -> None:
        """Move to the next entity."""
        await self.submit_current_judgement()

        if self.current_entity < len(self.samples) - 1:
            await self.load_entity(self.current_entity + 1)
        else:
            # All done
            await self.action_quit()

    async def action_previous_entity(self) -> None:
        """Move to the previous entity."""
        if self.current_entity > 0:
            await self.load_entity(self.current_entity - 1)

    async def action_clear_assignments(self) -> None:
        """Clear all group assignments for current entity."""
        table = self.query_one(RecordTable)
        table.group_assignments.clear()
        self.update_status_bar()

    async def action_show_help(self) -> None:
        """Show the help modal."""
        self.push_screen(HelpModal())

    async def action_jump_to_entity(self) -> None:
        """Jump to a specific entity number."""
        # TODO: Implement entity jump dialog
        pass

    async def submit_current_judgement(self) -> None:
        """Submit the current entity's judgement to the server."""
        if self.current_df is None or not self.current_cluster_id:
            return

        table = self.query_one(RecordTable)
        leaf_ids = self.current_df.select("leaf").to_series().to_list()
        endorsed_groups = table.get_judgement_groups(leaf_ids)

        judgement = Judgement(
            shown=self.current_cluster_id,
            endorsed=endorsed_groups,
            user_id=self.user_id,
        )

        try:
            _handler.send_eval_judgement(judgement=judgement)
        except Exception:
            # TODO: Better error handling
            pass

    async def action_quit(self) -> None:
        """Quit the application."""
        await self.submit_current_judgement()
        self.exit()


if __name__ == "__main__":
    app = EntityResolutionApp()
    app.run()
