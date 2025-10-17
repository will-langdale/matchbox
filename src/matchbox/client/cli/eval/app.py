"""Main application for entity resolution evaluation."""

import logging
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.cli.eval.handlers import EvaluationHandlers
from matchbox.client.cli.eval.state import EvaluationState
from matchbox.client.cli.eval.widgets.status import StatusBar
from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable
from matchbox.client.dags import DAG
from matchbox.client.eval import EvaluationItem, get_samples
from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.exceptions import MatchboxClientSettingsException

logger = logging.getLogger(__name__)


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
        ("ctrl+q,ctrl+c", "quit", "Quit"),
    ]

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
            dag: Pre-loaded DAG with warehouse location attached (optional for testing)
        """
        super().__init__()

        # Create single centralised state
        self.state = EvaluationState()

        # Set app reference for timer management
        self.state._app_ref = self

        # Initialise state with provided parameters
        self.state.resolution = resolution
        self.state.sample_limit = num_samples
        self.state.user_name = user or ""
        self.state.dag = dag

        # Create handlers for input and actions
        self.handlers = EvaluationHandlers(self)

    async def on_mount(self) -> None:
        """Initialise the application."""
        await self.authenticate()
        # DAG should already be loaded by CLI, but verify
        if self.state.dag is None:
            raise RuntimeError(
                "DAG not loaded. EntityResolutionApp requires a pre-loaded DAG. "
                "Ensure DAG is passed during initialisation."
            )
        await self.load_samples()

        # Check if no samples were loaded
        if self.state.queue.total_count == 0:
            self.state.has_no_samples = True
            self.state.update_status("◯ No data", "yellow")
            logger.warning(
                f"No samples available for resolution '{self.state.resolution}'. "
                "This may be because all clusters have been recently judged "
                "by this user, or the resolution has no probability data."
            )
            await self.handlers.action_show_no_samples()
            return

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
        if not samples_dict:
            return

        added = self.state.add_queue_items(list(samples_dict.values()))
        logger.info(
            "Loaded %s evaluation items (queue now %s/%s)",
            added,
            self.state.queue.total_count,
            self.state.sample_limit,
        )

    async def refresh_display(self) -> None:
        """Refresh display with current queue item."""
        current = self.state.queue.current
        if current:
            # Use the display columns from EvaluationItem
            self.state.set_display_data(current.display_columns)
        else:
            # No current item, clear display
            self.state.clear_display_data()

    def compose(self) -> ComposeResult:
        """Compose the main application UI."""
        yield Header()
        yield Vertical(
            StatusBar(self.state, classes="status-bar", id="status-bar"),
            ComparisonDisplayTable(self.state, id="record-table"),
            id="main-container",
        )
        yield Footer()

    async def on_key(self, event: events.Key) -> None:
        """Delegate key handling to handlers."""
        await self.handlers.handle_key_input(event)

    # Action methods - delegate to handlers
    async def action_next_entity(self) -> None:
        """Move to the next entity via queue rotation."""
        await self.handlers.action_next_entity()

    async def action_previous_entity(self) -> None:
        """Move to the previous entity via queue rotation."""
        await self.handlers.action_previous_entity()

    async def action_clear_assignments(self) -> None:
        """Clear all group assignments and held letters for current entity."""
        await self.handlers.action_clear_assignments()

    async def action_show_help(self) -> None:
        """Show the help modal."""
        await self.handlers.action_show_help()

    async def action_submit_and_fetch(self) -> None:
        """Submit all painted entities, remove from queue, fetch new samples."""
        await self.handlers.action_submit_and_fetch()

    async def _fetch_additional_samples(
        self, count: int
    ) -> dict[int, EvaluationItem] | None:
        """Fetch additional samples using the loaded DAG."""
        try:
            return get_samples(
                n=count,
                resolution=self.state.resolution,
                user_id=self.state.user_id,
                dag=self.state.dag,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to fetch samples: {type(e).__name__}: {e}")
            return None

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
