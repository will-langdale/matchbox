"""Textual-based entity resolution evaluation tool."""

from collections import deque
from pathlib import Path
from textwrap import dedent
from typing import Callable

import polars as pl
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Static,
)

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.cli.eval.utils import EvaluationItem, get_samples
from matchbox.common.exceptions import MatchboxClientSettingsException
from matchbox.common.graph import DEFAULT_RESOLUTION, ModelResolutionName


class EvaluationQueue:
    """Deque-based queue that maintains position illusion."""

    def __init__(self):
        """Initialize the queue."""
        self.items: deque[EvaluationItem] = deque()
        self._position_offset: int = 0  # Tracks "virtual position"

    @property
    def current(self) -> EvaluationItem | None:
        """Get the current item (always at index 0)."""
        return self.items[0] if self.items else None

    @property
    def current_position(self) -> int:
        """Current position for display (1-based)."""
        return self._position_offset + 1 if self.items else 0

    @property
    def total_count(self) -> int:
        """Total number of items in queue."""
        return len(self.items)

    @property
    def painted_items(self) -> list[EvaluationItem]:
        """Get all currently painted items in the queue."""
        return [item for item in self.items if item.is_painted]

    @property
    def painted_count(self) -> int:
        """Count of painted items ready for submission."""
        return len(self.painted_items)

    def move_next(self):
        """Rotate forward, increment position."""
        if len(self.items) > 1:
            self.items.append(self.items.popleft())
            self._position_offset = (self._position_offset + 1) % len(self.items)

    def move_previous(self):
        """Rotate backward, decrement position."""
        if len(self.items) > 1:
            self.items.appendleft(self.items.pop())
            self._position_offset = (self._position_offset - 1) % len(self.items)

    def submit_all_painted(self) -> list[EvaluationItem]:
        """Remove all painted items permanently."""
        painted = self.painted_items
        self.items = deque([item for item in self.items if not item.is_painted])
        # Reset position offset if we removed items
        if painted and self.items:
            self._position_offset = 0
        return painted

    def add_items(self, items: list[EvaluationItem]):
        """Add new items to the end of the queue."""
        self.items.extend(items)

    def clear(self):
        """Clear the entire queue."""
        self.items.clear()
        self._position_offset = 0


class EvaluationState:
    """Single source of truth for all application state."""

    def __init__(self):
        """Initialise evaluation state."""
        # Queue Management - replaces samples dict and entity_judgements
        self.queue: EvaluationQueue = EvaluationQueue()
        self.sample_limit: int = 100

        # UI State
        self.current_group_selection: str = ""  # Currently selected group letter

        # Display State (derived from current queue item)
        self.field_names: list[str] = []
        self.data_matrix: list[list[str]] = []
        self.leaf_ids: list[int] = []

        # User/Connection State
        self.user_name: str = ""
        self.user_id: int | None = None
        self.resolution: str = ""
        self.warehouse: str | None = None

        # Status/Feedback State
        self.status_message: str = ""
        self.is_submitting: bool = False

        # Observer pattern for view updates
        self.listeners: list[Callable] = []

        # Number key mapping
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

    @property
    def painted_count(self) -> int:
        """Get count of painted entities."""
        return self.queue.painted_count

    @property
    def current_cluster_id(self) -> int | None:
        """Get current cluster ID."""
        current = self.queue.current
        return current.cluster_id if current else None

    @property
    def current_df(self) -> pl.DataFrame | None:
        """Get current DataFrame."""
        current = self.queue.current
        return current.dataframe if current else None

    @property
    def current_assignments(self) -> dict[int, str]:
        """Get current entity's group assignments."""
        current = self.queue.current
        return current.assignments if current else {}

    def set_group_selection(self, group: str) -> None:
        """Set the current active group."""
        if group.isalpha() and len(group) == 1:
            self.current_group_selection = group.lower()
            self._notify_listeners()

    def clear_group_selection(self) -> None:
        """Clear the current group selection."""
        self.current_group_selection = ""
        self._notify_listeners()

    def assign_column_to_group(self, column_number: int, group: str) -> None:
        """Assign a column to a group."""
        col_index = column_number - 1
        current = self.queue.current
        if current and 0 <= col_index < len(self.leaf_ids):
            current.assignments[col_index] = group
            self._notify_listeners()

    def clear_current_assignments(self) -> None:
        """Clear all group assignments for current entity."""
        current = self.queue.current
        if current:
            current.assignments.clear()
        self._notify_listeners()

    def set_display_data(
        self, field_names: list[str], data_matrix: list[list[str]], leaf_ids: list[int]
    ) -> None:
        """Set the display data."""
        self.field_names = field_names
        self.data_matrix = data_matrix
        self.leaf_ids = leaf_ids
        self._notify_listeners()

    def clear_display_data(self) -> None:
        """Clear all display data."""
        self.field_names = []
        self.data_matrix = []
        self.leaf_ids = []
        self._notify_listeners()

    def get_group_counts(self) -> dict[str, int]:
        """Get count of columns in each group for current entity."""
        assignments = self.current_assignments
        counts = {}
        for group in assignments.values():
            counts[group] = counts.get(group, 0) + 1

        # Include unassigned count if there are unassigned columns
        assigned_count = sum(counts.values())
        remaining_count = len(self.leaf_ids) - assigned_count
        if remaining_count > 0:
            counts["unassigned"] = remaining_count

        return counts

    def get_judgement_groups(self) -> list[list[int]]:
        """Convert current assignments to judgement format."""
        current = self.queue.current
        if current and self.user_id:
            judgement = current.to_judgement(self.user_id)
            return judgement.endorsed
        return []

    def parse_number_key(self, key: str) -> int | None:
        """Convert number key to column number."""
        return self.number_keys.get(key)

    def has_current_assignments(self) -> bool:
        """Check if current entity has any group assignments."""
        return len(self.current_assignments) > 0

    def update_status(self, message: str) -> None:
        """Update status message."""
        self.status_message = message
        self._notify_listeners()

    def clear_status(self) -> None:
        """Clear status message."""
        self.status_message = ""
        self._notify_listeners()

    def add_listener(self, callback: Callable) -> None:
        """Add a callback to be notified when state changes."""
        self.listeners.append(callback)

    def _notify_listeners(self) -> None:
        """Notify all listeners of state changes."""
        for callback in self.listeners:
            try:
                callback()
            except Exception:
                # Don't let listener errors crash the UI
                pass


class ComparisonDisplayTable(Widget):
    """Minimal table for side-by-side record comparison with colour highlighting."""

    def __init__(self, state: EvaluationState, **kwargs):
        """Initialise the comparison display table."""
        super().__init__(**kwargs)
        self.state = state
        self.state.add_listener(self._on_state_change)

    def render(self) -> Table:
        """Render the table with current state using Rich Table."""
        if not self.state.field_names or not self.state.data_matrix:
            # Create a simple loading table
            loading_table = Table(show_header=False, show_lines=False)
            loading_table.add_column("")
            loading_table.add_row("Loading...")
            return loading_table

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
        table.add_column("Field", style="bright_white", min_width=20, max_width=35)

        # Add columns for each record with group styling
        current_assignments = self.state.current_assignments
        for col_idx in range(len(self.state.leaf_ids)):
            col_num = col_idx + 1

            if col_idx in current_assignments:
                group = current_assignments[col_idx]
                colour, symbol = GroupStyler.get_style(group)
                header = f"[{colour}]{symbol} {col_num}[/]"
                # Use group colour for the entire column
                table.add_column(header, style=colour, min_width=15, max_width=35)
            else:
                table.add_column(str(col_num), style="dim", min_width=15, max_width=35)

        # Add data rows
        for row_idx, field_name in enumerate(self.state.field_names):
            if field_name == "---":
                # Add minimal separator row
                separator_row = ["─" * 15] + ["─" * 8] * len(self.state.leaf_ids)
                table.add_row(*separator_row, style="dim")
                continue

            # Prepare row data
            row_data = [field_name]

            if row_idx < len(self.state.data_matrix):
                for value in self.state.data_matrix[row_idx]:
                    cell_value = str(value) if value else ""
                    # Show full data - no truncation
                    row_data.append(cell_value)
            else:
                # Fill with empty values if data is missing
                row_data.extend([""] * len(self.state.leaf_ids))

            table.add_row(*row_data)

        return table

    def _on_state_change(self) -> None:
        """Handle state changes by refreshing the display."""
        self.refresh()


class GroupStyler:
    """Generate consistent colours and symbols with cycling to minimise duplicates."""

    # High contrast colours distributed to avoid similar adjacents
    COLOURS = [
        "red",
        "blue",
        "green",
        "yellow",
        "magenta",
        "cyan",
        "bright_red",
        "bright_green",
        "bright_blue",
        "bright_yellow",
        "bright_magenta",
        "bright_cyan",
        "white",
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

    # Class-level tracking for consistent assignments
    _group_styles = {}  # group_name -> (colour, symbol)
    _used_colours = set()
    _used_symbols = set()
    _colour_index = 0
    _symbol_index = 0

    @classmethod
    def get_style(cls, group_name: str) -> tuple[str, str]:
        """Get consistent colour and symbol for a group name using cycling."""
        # Return cached style if already assigned
        if group_name in cls._group_styles:
            return cls._group_styles[group_name]

        # Assign next available colour
        colour = cls._get_next_colour()
        symbol = cls._get_next_symbol()

        # Cache the assignment
        cls._group_styles[group_name] = (colour, symbol)
        return colour, symbol

    @classmethod
    def _get_next_colour(cls) -> str:
        """Get the next colour in cycle, avoiding duplicates when possible."""
        # If we haven't used all colours yet, find an unused one
        if len(cls._used_colours) < len(cls.COLOURS):
            while cls.COLOURS[cls._colour_index] in cls._used_colours:
                cls._colour_index = (cls._colour_index + 1) % len(cls.COLOURS)

        colour = cls.COLOURS[cls._colour_index]
        cls._used_colours.add(colour)
        cls._colour_index = (cls._colour_index + 1) % len(cls.COLOURS)

        return colour

    @classmethod
    def _get_next_symbol(cls) -> str:
        """Get the next symbol in cycle, avoiding duplicates when possible."""
        # If we haven't used all symbols yet, find an unused one
        if len(cls._used_symbols) < len(cls.SYMBOLS):
            while cls.SYMBOLS[cls._symbol_index] in cls._used_symbols:
                cls._symbol_index = (cls._symbol_index + 1) % len(cls.SYMBOLS)

        symbol = cls.SYMBOLS[cls._symbol_index]
        cls._used_symbols.add(symbol)
        cls._symbol_index = (cls._symbol_index + 1) % len(cls.SYMBOLS)

        return symbol

    @classmethod
    def get_display_text(cls, group_name: str, count: int) -> tuple[str, str]:
        """Get formatted display text with colour and symbol."""
        colour, symbol = cls.get_style(group_name)
        text = f"{symbol} {group_name.upper()} ({count})"
        return text, colour

    @classmethod
    def reset(cls):
        """Reset all assignments (useful for testing)."""
        cls._group_styles.clear()
        cls._used_colours.clear()
        cls._used_symbols.clear()
        cls._colour_index = 0
        cls._symbol_index = 0


class StatusBar(Widget):
    """Widget to display group assignments with coloured indicators."""

    def __init__(self, state: EvaluationState, **kwargs):
        """Initialise the status bar."""
        super().__init__(**kwargs)
        self.state = state
        self.state.add_listener(self._on_state_change)

    def render(self) -> Text:
        """Render group status with coloured symbols and current selection."""
        text = Text()

        # Show status message if any
        if self.state.status_message:
            text.append(self.state.status_message, style="bright_yellow")
            return text

        # Show current selection prominently if any
        current_group = self.state.current_group_selection
        if current_group:
            colour, symbol = GroupStyler.get_style(current_group)
            text.append("Selected: ", style="bright_white")
            text.append(
                f"{symbol} {current_group.upper()}", style=f"bold {colour} on black"
            )
            text.append(" | ", style="dim")

        # Show group assignments
        groups = self.state.get_group_counts()

        if not groups:
            text.append("No groups assigned", style="dim")
        else:
            text.append("Groups: ", style="bright_white")
            for i, (group, count) in enumerate(groups.items()):
                if i > 0:
                    text.append("  ")

                # Use GroupStyler for consistent colour and symbol
                display_text, colour = GroupStyler.get_display_text(group, count)
                text.append(display_text, style=f"bold {colour}")

        # Add helpful instruction if no group selected
        if not current_group:
            text.append(" | ", style="dim")
            text.append("Press letter to select group", style="yellow")

        return text

    def _on_state_change(self) -> None:
        """Handle state changes by refreshing the display."""
        self.refresh()


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
                    • Ctrl+Q - Quit

                    Visual feedback:
                    • Records are columns, fields are rows
                    • Same field types grouped together for easy comparison
                    • Each group gets unique colour + symbol combination
                    • Column headers show group assignment with coloured symbols
                    • Status bar shows group counts with visual indicators

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
        ("space", "submit_and_fetch", "Submit & fetch more"),
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
        """Initialise the entity resolution app."""
        super().__init__()

        # Create single centralised state
        self.state = EvaluationState()

        # Initialise state with provided parameters
        self.state.resolution = resolution
        self.state.sample_limit = num_samples
        self.state.user_name = user or ""
        self.state.warehouse = warehouse

        # Add listener to update header when state changes
        self.state.add_listener(self.update_navigation_header)

    async def on_mount(self) -> None:
        """Initialise the application."""
        await self.authenticate()
        await self.load_samples()
        if self.state.queue.current:
            await self.refresh_display()

    async def authenticate(self) -> None:
        """Authenticate with the server."""
        # Use injected user or fall back to settings
        user_name = self.state.user_name or settings.user
        if not user_name:
            raise MatchboxClientSettingsException("User name is unset.")

        self.state.user_name = user_name
        self.state.user_id = _handler.login(user_name=user_name)

    async def load_samples(self) -> None:
        """Load evaluation samples from the server."""
        samples_dict = await self._fetch_additional_samples(self.state.sample_limit)
        if samples_dict:
            # samples_dict now contains EvaluationItems, not DataFrames
            self.state.queue.add_items(list(samples_dict.values()))

    async def refresh_display(self) -> None:
        """Refresh display with current queue item."""
        current = self.state.queue.current
        if current:
            # Use the pre-processed field data from EvaluationItem
            self.state.set_display_data(
                current.field_names, current.data_matrix, current.leaf_ids
            )
        else:
            # No current item, clear display
            self.state.clear_display_data()

        self.update_navigation_header()

    def compose(self) -> ComposeResult:
        """Compose the main application UI."""
        yield Header()
        yield Static("Loading...", classes="navigation-header", id="nav-header")
        yield Container(
            StatusBar(self.state, classes="status-bar", id="status-bar"),
            ComparisonDisplayTable(self.state, id="record-table"),
            Static(
                "Press any letter (a-z) to select group, "
                "then numbers to assign columns",
                classes="command-area",
                id="help-text",
            ),
            id="main-container",
        )
        yield Footer()

    def update_navigation_header(self) -> None:
        """Update the navigation header with current progress."""
        total = self.state.queue.total_count
        current = self.state.queue.current_position
        painted_count = self.state.painted_count

        # Show painted entities count
        painted_text = f" | ■ {painted_count} painted" if painted_count > 0 else ""

        # Mark current entity as painted if it has assignments
        current_painted = "■ " if self.state.has_current_assignments() else ""

        nav_text = (
            f"{current_painted}Entity {current}/{total}{painted_text} | → Next | "
            f"← Previous | Space Submit | ? Help"
        )

        header = self.query_one("#nav-header", Static)
        header.update(nav_text)

    def on_key(self, event) -> None:
        """Handle keyboard events for group assignment shortcuts."""
        key = event.key

        # Handle navigation keys first - let the normal key binding system handle these
        if key in ["left", "right", "enter", "space"]:
            return

        # Handle special keys
        if key == "escape":
            # Clear current group selection
            self.state.clear_group_selection()
            self.update_help_text()
            event.prevent_default()
            return

        # Handle letter key presses (set current group)
        if key.isalpha() and len(key) == 1:
            self.state.set_group_selection(key)
            self.update_help_text()
            event.prevent_default()
            return

        # Handle number key presses (assign columns to current group)
        column_number = self.state.parse_number_key(key)
        if column_number is not None:
            current_group = self.state.current_group_selection
            if current_group:  # Only assign if we have a group selected
                if 1 <= column_number <= len(self.state.leaf_ids):
                    self.state.assign_column_to_group(column_number, current_group)
            event.prevent_default()
            return

    def update_help_text(self) -> None:
        """Update the help text to show current group selection."""
        current_group = self.state.current_group_selection
        if current_group:
            colour, symbol = GroupStyler.get_style(current_group)
            help_text = (
                f"Group [{colour}]{symbol} {current_group.upper()}[/] selected - "
                "press numbers to assign columns"
            )
        else:
            help_text = (
                "Press any letter (a-z) to select group, then numbers to assign columns"
            )

        help_widget = self.query_one("#help-text", Static)
        help_widget.update(help_text)

    async def action_next_entity(self) -> None:
        """Move to the next entity via queue rotation."""
        if self.state.queue.total_count > 1:
            self.state.queue.move_next()
            await self.refresh_display()
        else:
            # All done - auto-submit everything and quit
            await self.action_submit_and_fetch()
            await self.action_quit()

    async def action_previous_entity(self) -> None:
        """Move to the previous entity via queue rotation."""
        if self.state.queue.total_count > 1:
            self.state.queue.move_previous()
            await self.refresh_display()

    async def action_clear_assignments(self) -> None:
        """Clear all group assignments and held letters for current entity."""
        self.state.clear_current_assignments()
        self.state.clear_group_selection()
        self.update_help_text()

    async def action_show_help(self) -> None:
        """Show the help modal."""
        self.push_screen(HelpModal())

    async def action_submit_and_fetch(self) -> None:
        """Submit all painted entities, remove from queue, fetch new samples."""
        painted_items = self.state.queue.painted_items
        painted_count = len(painted_items)

        if painted_count == 0:
            self.state.update_status("Nothing to submit - no fully painted entities")
            self.set_timer(2.0, self.state.clear_status)
            return

        # Update status to show we're submitting
        self.state.is_submitting = True
        self.state.update_status(f"✓ Submitting {painted_count} painted entities...")

        # Submit each painted item
        successful_submissions = 0
        for item in painted_items:
            try:
                judgement = item.to_judgement(self.state.user_id)
                _handler.submit_judgement(
                    judgement=judgement, user_id=self.state.user_id
                )
                successful_submissions += 1
            except Exception:
                # Continue with other submissions if one fails
                continue

        # Remove all painted items from queue (they're done forever)
        self.state.queue.submit_all_painted()

        # Update status to show completion
        remaining_count = self.state.queue.total_count
        self.state.update_status(
            f"✓ Submitted {successful_submissions}/{painted_count} entities, "
            f"removed from queue ({remaining_count} remaining)"
        )

        # Refresh display to show current entity (queue auto-advances)
        await self.refresh_display()

        # Fetch new samples to backfill the queue
        await self._backfill_samples()

        self.state.is_submitting = False

        # Show final status
        final_count = self.state.queue.total_count
        if final_count > remaining_count:
            self.state.update_status(
                f"✓ Queue ready - {final_count} entities available"
            )
        else:
            self.state.update_status(
                f"✓ Submission complete - {final_count} entities in queue"
            )

        # Clear status after a longer delay to show the result
        self.set_timer(4.0, self.state.clear_status)

    async def _backfill_samples(self) -> None:
        """Fetch new samples to replace submitted ones."""
        try:
            current_count = self.state.queue.total_count
            desired_count = self.state.sample_limit
            needed = max(0, desired_count - current_count)

            if needed > 0:
                self.state.update_status(f"Fetching {needed} replacement samples...")

                # Fetch replacement samples
                new_samples_dict = await self._fetch_additional_samples(needed)

                if new_samples_dict and len(new_samples_dict) > 0:
                    # new_samples_dict now contains EvaluationItems
                    new_items = list(new_samples_dict.values())
                    self.state.queue.add_items(new_items)

                    # Update displays and status
                    self.update_navigation_header()
                    self.state.update_status(
                        f"Queue updated: added {len(new_items)} new samples"
                    )

                    # If we're currently viewing an empty state, refresh display
                    if not self.state.current_df and len(new_items) > 0:
                        await self.refresh_display()
                else:
                    self.state.update_status(
                        "No new samples available - queue exhausted"
                    )
            else:
                self.state.update_status("Queue is at capacity")

        except Exception as e:
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            self.state.update_status(f"Error fetching samples: {error_msg}")

    async def _fetch_additional_samples(
        self, count: int
    ) -> dict[int, EvaluationItem] | None:
        """Fetch additional samples from the server."""
        try:
            # Temporarily patch warehouse setting if provided
            original_warehouse = None
            if self.state.warehouse:
                original_warehouse = getattr(settings, "default_warehouse", None)
                settings.default_warehouse = self.state.warehouse

            try:
                return get_samples(
                    n=count,
                    resolution=self.state.resolution,
                    user_id=self.state.user_id,
                    clients={},
                    use_default_client=True,
                )
            finally:
                # Restore original warehouse setting
                if self.state.warehouse:
                    if original_warehouse:
                        settings.default_warehouse = original_warehouse
                    elif hasattr(settings, "default_warehouse"):
                        delattr(settings, "default_warehouse")
        except Exception:
            return None

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


if __name__ == "__main__":
    app = EntityResolutionApp()
    app.run()
