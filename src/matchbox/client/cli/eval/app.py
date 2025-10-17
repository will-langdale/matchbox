"""Main application for entity resolution evaluation."""

import logging
from collections import deque
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Footer, Header

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.cli.eval.modals import HelpModal, NoSamplesModal
from matchbox.client.cli.eval.widgets.status import StatusBar
from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable
from matchbox.client.dags import DAG
from matchbox.client.eval import EvaluationItem, create_judgement, get_samples
from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.exceptions import MatchboxClientSettingsException

logger = logging.getLogger(__name__)


class EvaluationQueue:
    """Deque-based queue with current item always at front."""

    def __init__(self) -> None:
        """Initialise the queue."""
        self.items: deque[EvaluationItem] = deque()

    @property
    def current(self) -> EvaluationItem | None:
        """Get the current item (always at index 0)."""
        return self.items[0] if self.items else None

    @property
    def total_count(self) -> int:
        """Total number of items in queue."""
        return len(self.items)

    def skip_current(self) -> None:
        """Move current to back of queue."""
        if len(self.items) > 1:
            self.items.rotate(-1)

    def remove_current(self) -> EvaluationItem | None:
        """Remove and return current item."""
        return self.items.popleft() if self.items else None

    def add_items(self, items: list[EvaluationItem]) -> int:
        """Add new items to queue, preventing duplicates.

        Returns:
            Number of unique items added.
        """
        if not items:
            return 0

        existing_ids = {item.cluster_id for item in self.items}
        unique_items = [item for item in items if item.cluster_id not in existing_ids]

        if unique_items:
            self.items.extend(unique_items)

        return len(unique_items)

    def remove_by_cluster_ids(self, cluster_ids: set[int]) -> None:
        """Remove items with cluster IDs in the provided set."""
        if not cluster_ids:
            return

        self.items = deque(
            item for item in self.items if item.cluster_id not in cluster_ids
        )


class EntityResolutionApp(App):
    """Main Textual application for entity resolution evaluation."""

    CSS_PATH = Path(__file__).parent / "styles.css"

    BINDINGS = [
        ("right", "skip", "Skip → back"),
        ("space", "submit", "Submit current"),
        ("escape", "clear", "Clear groups"),
        ("question_mark,f1", "show_help", "Help"),
        ("ctrl+q,ctrl+c", "quit", "Quit"),
    ]

    # Queue (inline class)
    queue: EvaluationQueue

    current_group: reactive[str] = reactive("")
    assignments: reactive[dict[int, str]] = reactive({}, init=False)

    status_message: reactive[str] = reactive("○ Ready")
    status_colour: reactive[str] = reactive("dim")

    sample_limit: int = 100
    resolution: ModelResolutionPath = ""
    user_id: int | None = None
    user_name: str = ""
    dag: DAG | None = None
    has_no_samples: bool = False

    _status_timer: Timer | None = None
    number_keys = {
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

    def __init__(
        self,
        resolution: ModelResolutionPath,
        num_samples: int = 100,
        user: str | None = None,
        dag: DAG | None = None,
    ) -> None:
        """Initialise the entity resolution app.

        Args:
            resolution: The model resolution to evaluate
            num_samples: Number of clusters to sample for evaluation
            user: Username for authentication (overrides settings)
            dag: Pre-loaded DAG with warehouse location attached
        """
        super().__init__()

        self.queue = EvaluationQueue()
        self.resolution = resolution
        self.sample_limit = num_samples
        self.user_name = user or ""
        self.dag = dag

    async def on_mount(self) -> None:
        """Initialise the application."""
        await self.authenticate()

        if self.dag is None:
            raise RuntimeError(
                "DAG not loaded. EntityResolutionApp requires a pre-loaded DAG."
            )

        await self.load_samples()

        if self.queue.total_count == 0:
            self.has_no_samples = True
            self.update_status("◯ No data", "yellow")
            logger.warning(f"No samples available for resolution '{self.resolution}'.")
            await self.action_show_no_samples()
            return

        self.refresh_display()

    async def authenticate(self) -> None:
        """Authenticate with the server."""
        user_name = self.user_name or settings.user
        if not user_name:
            raise MatchboxClientSettingsException("User name is unset.")

        self.user_name = user_name
        self.user_id = _handler.login(user_name=user_name)

    async def load_samples(self) -> None:
        """Load evaluation samples from the server."""
        samples_dict = await self._fetch_additional_samples(self.sample_limit)
        if not samples_dict:
            return

        added = self.queue.add_items(list(samples_dict.values()))
        logger.info(
            "Loaded %s evaluation items (queue now %s/%s)",
            added,
            self.queue.total_count,
            self.sample_limit,
        )

    def refresh_display(self) -> None:
        """Refresh display with current queue item."""
        current = self.queue.current
        if current:
            self.assignments = {}
            self.current_group = ""

            table = self.query_one(ComparisonDisplayTable)
            table.load_comparison(current)

            status_bar = self.query_one(StatusBar)
            status_bar.queue_position = 1
            status_bar.queue_total = self.queue.total_count
            status_bar.current_group = self.current_group

    def compose(self) -> ComposeResult:
        """Compose the main application UI."""
        yield Header()
        yield Vertical(
            StatusBar(id="status-bar", classes="status-bar"),
            ComparisonDisplayTable(id="record-table"),
            id="main-container",
        )
        yield Footer()

    async def on_key(self, event: events.Key) -> None:
        """Handle keyboard shortcuts for group assignment."""
        key = event.key

        if key.isalpha() and len(key) == 1:
            self.current_group = key.lower()
            status_bar = self.query_one(StatusBar)
            status_bar.current_group = self.current_group
            event.stop()
            return

        if key in self.number_keys and self.current_group:
            col_num = self.number_keys[key]
            current = self.queue.current
            if current and col_num <= len(current.display_columns):
                new_assignments = self.assignments.copy()
                new_assignments[col_num - 1] = self.current_group
                self.assignments = new_assignments

                current.assignments[col_num - 1] = self.current_group

                table = self.query_one(ComparisonDisplayTable)
                table.refresh()

                status_bar = self.query_one(StatusBar)
                status_bar.group_counts = self._compute_group_counts()

            event.stop()
            return

    def _compute_group_counts(self) -> dict[str, int]:
        """Compute group counts for current entity."""
        current = self.queue.current
        if not current:
            return {}

        counts = {}

        for display_col_index, group in current.assignments.items():
            if display_col_index < len(current.duplicate_groups):
                duplicate_group_size = len(current.duplicate_groups[display_col_index])
                counts[group] = counts.get(group, 0) + duplicate_group_size

        if self.current_group and self.current_group not in counts:
            counts[self.current_group] = 0

        assigned_display_cols = set(current.assignments.keys())
        unassigned_leaf_count = 0
        for display_col_index in range(len(current.duplicate_groups)):
            if display_col_index not in assigned_display_cols:
                duplicate_group = current.duplicate_groups[display_col_index]
                unassigned_leaf_count += len(duplicate_group)

        if unassigned_leaf_count > 0:
            counts["unassigned"] = unassigned_leaf_count

        return counts

    async def action_skip(self) -> None:
        """Skip current entity (moves to back of queue)."""
        if self.queue.total_count > 1:
            self.queue.skip_current()
            self.refresh_display()
        else:
            await self.action_submit()
            if self.queue.total_count == 0:
                self.exit()

    async def action_submit(self) -> None:
        """Submit current entity if fully painted."""
        current = self.queue.current

        if not current:
            return

        # Check if all columns are assigned
        is_painted = len(current.assignments) == len(current.display_columns)
        if not is_painted:
            self.update_status("⚠ Incomplete", "yellow", clear_after=3.0)
            return

        try:
            if self.user_id is None:
                raise RuntimeError("User ID is not set")
            judgement = create_judgement(current, self.user_id)
            _handler.send_eval_judgement(judgement)
        except Exception as exc:
            self.update_status("⚠ Send failed", "red", clear_after=4.0)
            logger.exception(f"Failed to submit: {exc}")
            return

        self.queue.remove_current()
        await self._fetch_more_samples()
        self.refresh_display()
        self.update_status("✓ Sent", "green", clear_after=2.0)

    async def action_clear(self) -> None:
        """Clear current entity's group assignments."""
        current = self.queue.current
        if current:
            current.assignments.clear()
            self.assignments = {}
            self.current_group = ""

            table = self.query_one(ComparisonDisplayTable)
            table.refresh()

            status_bar = self.query_one(StatusBar)
            status_bar.current_group = ""
            status_bar.group_counts = {}

    async def action_show_help(self) -> None:
        """Show the help modal."""
        self.push_screen(HelpModal())

    async def action_show_no_samples(self) -> None:
        """Show the no samples modal."""
        self.push_screen(NoSamplesModal())

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    async def _fetch_more_samples(self) -> None:
        """Fetch new samples to backfill queue."""
        current_count = self.queue.total_count
        desired_count = self.sample_limit
        needed = max(0, desired_count - current_count)

        if needed <= 0:
            return

        self.update_status("⚡ Fetching", "yellow")
        logger.info(
            "Backfilling queue: need %s samples to reach limit of %s",
            needed,
            desired_count,
        )

        remaining = needed
        total_added = 0

        while remaining > 0:
            new_samples_dict = await self._fetch_additional_samples(remaining)
            if not new_samples_dict:
                break

            new_items = list(new_samples_dict.values())
            added = self.queue.add_items(new_items)
            total_added += added

            if added == 0:
                # Backend returned only duplicates
                break

            remaining = max(0, desired_count - self.queue.total_count)

        if total_added == 0 and self.queue.total_count == 0:
            self.has_no_samples = True
            self.update_status("◯ No data", "yellow")
            await self.action_show_no_samples()
        else:
            self.update_status("✓ Ready", "green")

    async def _fetch_additional_samples(
        self, count: int
    ) -> dict[int, EvaluationItem] | None:
        """Fetch additional samples using the loaded DAG."""
        try:
            return get_samples(
                n=count,
                resolution=self.resolution,
                user_id=self.user_id,
                dag=self.dag,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to fetch samples: {type(e).__name__}: {e}")
            return None

    def update_status(
        self,
        message: str,
        colour: str = "dim",
        clear_after: float | None = None,
    ) -> None:
        """Update status message with optional auto-clear.

        Args:
            message: Status message to display
            colour: Colour for the message
            clear_after: Seconds after which to auto-clear
        """
        if self._status_timer:
            self._status_timer.stop()

        self.status_message = message
        self.status_colour = colour

        status_bar = self.query_one(StatusBar)
        status_bar.status_message = message
        status_bar.status_colour = colour

        if clear_after:
            self._status_timer = self.set_timer(
                clear_after, lambda: self.update_status("○ Ready", "dim")
            )
