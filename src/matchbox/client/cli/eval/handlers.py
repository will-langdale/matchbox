"""Input handlers and action methods for entity resolution evaluation."""

import logging

from matchbox.client import _handler
from matchbox.client.cli.eval.modals import HelpModal

logger = logging.getLogger(__name__)


class EvaluationHandlers:
    """Handles all input events and actions for the evaluation app."""

    def __init__(self, app):
        """Initialise handlers with app reference."""
        self.app = app
        self.state = app.state

    async def handle_key_input(self, event) -> None:
        """Handle keyboard events for group assignment shortcuts."""
        key = event.key

        if key in ["left", "right", "enter", "space"]:
            return

        if key == "escape":
            self.state.clear_group_selection()
            event.prevent_default()
            return

        if key.isalpha() and len(key) == 1:
            self.state.set_group_selection(key)
            event.prevent_default()
            return

        column_number = self.state.parse_number_key(key)
        if column_number is not None:
            current_group = self.state.current_group_selection
            if current_group:
                current = self.state.queue.current
                if current and 1 <= column_number <= len(current.display_columns):
                    self.state.assign_column_to_group(column_number, current_group)
            event.prevent_default()

    async def action_next_entity(self) -> None:
        """Move to the next entity via queue rotation."""
        if self.state.queue.total_count > 1:
            self.state.queue.move_next()
            await self.app.refresh_display()
        else:
            await self.action_submit_and_fetch()
            await self.app.action_quit()

    async def action_previous_entity(self) -> None:
        """Move to the previous entity via queue rotation."""
        if self.state.queue.total_count > 1:
            self.state.queue.move_previous()
            await self.app.refresh_display()

    async def action_clear_assignments(self) -> None:
        """Clear all group assignments for current entity."""
        self.state.clear_current_assignments()
        self.state.clear_group_selection()

    async def action_show_help(self) -> None:
        """Show the help modal."""
        self.app.push_screen(HelpModal())

    async def action_submit_and_fetch(self) -> None:
        """Submit current entity if fully painted, then fetch more."""
        current = self.state.queue.current

        # Validate we have a current item
        if not current:
            self.state.update_status("◯ No data", "yellow")
            return

        # Validate current item is fully painted
        if not current.is_painted:
            self.state.update_status("⚠ Not ready", "yellow", auto_clear_after=3.0)
            return

        # Submit just the current item
        self.state.is_submitting = True
        self.state.update_status("⚡ Sending", "yellow")

        judgement = current.to_judgement(self.state.user_id)
        _handler.send_eval_judgement(judgement=judgement)

        logger.info(f"Successfully submitted entity {current.cluster_id}")

        # Remove current item from queue
        self.state.queue.remove_current()
        remaining = self.state.queue.total_count
        logger.info(f"Removed submitted item, {remaining} entities remaining in queue")

        # Backfill queue
        await self._backfill_samples()

        # Refresh display to show next item
        await self.app.refresh_display()

        self.state.is_submitting = False
        self.state.update_status("✓ Sent", "green", auto_clear_after=2.0)

    async def _backfill_samples(self) -> None:
        """Fetch new samples to replace submitted ones."""
        current_count = self.state.queue.total_count
        desired_count = self.state.sample_limit
        needed = max(0, desired_count - current_count)

        if needed <= 0:
            self.state.update_status("✓ Ready", "green")
            logger.info(f"Queue already at capacity: {current_count}/{desired_count}")
            return

        self.state.update_status("⚡ Fetching", "yellow")
        logger.info(
            "Backfilling queue: need %s samples to reach limit of %s",
            needed,
            desired_count,
        )

        new_samples_dict = await self.app._fetch_additional_samples(needed)
        if new_samples_dict:
            await self._process_new_samples(new_samples_dict)
        else:
            self._handle_no_samples_available(needed)

    async def _process_new_samples(self, new_samples_dict: dict) -> None:
        """Process newly fetched samples and update the queue."""
        new_items = list(new_samples_dict.values())
        self.state.queue.add_items(new_items)
        self.state.update_status("✓ Ready", "green")
        logger.info(f"Successfully added {len(new_items)} new samples to queue")

        if self.state.current_df is None and new_items:
            await self.app.refresh_display()

    def _handle_no_samples_available(self, needed: int) -> None:
        """Handle the case where no new samples are available."""
        # Check if queue is now completely empty
        if self.state.queue.total_count == 0:
            # Queue is empty and we can't fetch more - truly out of samples
            self.state.has_no_samples = True
            self.state.update_status("◯ No data", "yellow")
            logger.warning(
                f"Queue is empty and no new samples available - "
                f"requested {needed} but got none"
            )
        else:
            # Queue still has items, just can't fetch more right now
            self.state.update_status("◯ Empty", "dim")
            logger.warning(
                f"No new samples available - requested {needed} but got none"
            )
