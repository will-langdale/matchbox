"""Input handlers and action methods for entity resolution evaluation."""

import logging

from matchbox.client import _handler
from matchbox.client.cli.eval.modals import HelpModal, PlotModal
from matchbox.client.cli.eval.plot.data import (
    can_show_plot,
    refresh_judgements_for_plot,
)

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

        # Handle navigation keys first - let the normal key binding system handle these
        if key in ["left", "right", "enter", "space"]:
            return

        # Handle special keys
        if key == "escape":
            # Clear current group selection
            self.state.clear_group_selection()
            event.prevent_default()
            return

        # Handle letter key presses (set current group)
        if key.isalpha() and len(key) == 1:
            self.state.set_group_selection(key)
            event.prevent_default()
            return

        # Handle slash key (plot toggle)
        if key == "slash":
            await self.handle_plot_toggle()
            event.prevent_default()
            return

        # Handle number key presses (assign columns to current group)
        column_number = self.state.parse_number_key(key)
        if column_number is not None:
            current_group = self.state.current_group_selection
            if current_group:  # Only assign if we have a group selected
                current = self.state.queue.current
                if current and 1 <= column_number <= len(current.display_columns):
                    self.state.assign_column_to_group(column_number, current_group)
            event.prevent_default()
            return

    async def handle_plot_toggle(self) -> None:
        """Handle plot toggle with centralised data validation."""
        # Check if plot can be shown
        can_show, status_msg = can_show_plot(self.state)

        if not can_show:
            self.state.update_status(status_msg, "yellow", auto_clear_after=2.0)
            return

        # Update status to show we're refreshing
        self.state.update_status("⏳ Loading", "yellow")

        # Refresh judgements before showing plot
        success, refresh_status = refresh_judgements_for_plot(self.state)

        if not success:
            # Show error status
            self.state.update_status(refresh_status, "red", auto_clear_after=4.0)
            return

        # Show the status from refresh, then show modal
        self.state.update_status(refresh_status, "green", auto_clear_after=2.0)

        self.app.push_screen(PlotModal(self.state))

    async def action_next_entity(self) -> None:
        """Move to the next entity via queue rotation."""
        if self.state.queue.total_count > 1:
            self.state.queue.move_next()
            await self.app.refresh_display()
        else:
            # All done - auto-submit everything and quit
            await self.action_submit_and_fetch()
            await self.app.action_quit()

    async def action_previous_entity(self) -> None:
        """Move to the previous entity via queue rotation."""
        if self.state.queue.total_count > 1:
            self.state.queue.move_previous()
            await self.app.refresh_display()

    async def action_clear_assignments(self) -> None:
        """Clear all group assignments and held letters for current entity."""
        self.state.clear_current_assignments()
        self.state.clear_group_selection()

    async def action_toggle_view_mode(self) -> None:
        """Toggle between compact and detailed view modes."""
        self.state.toggle_view_mode()

    async def action_show_help(self) -> None:
        """Show the help modal."""
        self.app.push_screen(HelpModal())

    async def action_submit_and_fetch(self) -> None:
        """Submit all painted entities, remove from queue, fetch new samples."""
        painted_items = self.state.queue.painted_items
        painted_count = len(painted_items)

        if painted_count == 0:
            self.state.update_status("◯ Nothing", "dim", auto_clear_after=2.0)
            return

        # Update status to show we're submitting
        self.state.is_submitting = True
        self.state.update_status("⚡ Sending", "yellow")

        # Submit each painted item
        successful_submissions = 0
        for item in painted_items:
            judgement = item.to_judgement(self.state.user_id)
            _handler.send_eval_judgement(judgement=judgement)
            successful_submissions += 1

        # Remove all painted items from queue (they're done forever)
        self.state.queue.submit_all_painted()

        # Update status to show completion
        remaining_count = self.state.queue.total_count
        self.state.update_status("✓ Sent", "green")

        # Log detailed submission info for debugging
        logger.info(
            f"Successfully submitted {successful_submissions}/{painted_count} "
            "painted entities"
        )
        logger.info(
            f"Removed submitted items from queue, {remaining_count} entities remaining"
        )

        # Refresh display to show current entity (queue auto-advances)
        await self.app.refresh_display()

        # Fetch new samples to backfill the queue
        await self._backfill_samples()

        self.state.is_submitting = False

        # Show final status
        final_count = self.state.queue.total_count
        if final_count > remaining_count:
            self.state.update_status("✓ Ready", "green", auto_clear_after=4.0)
            # Log detailed backfill info
            logger.info(f"Queue backfilled: now has {final_count} entities available")
        else:
            self.state.update_status("✓ Done", "green", auto_clear_after=4.0)
            # Log completion info
            logger.info(
                f"Submission complete: {final_count} entities remaining in queue"
            )

    async def _backfill_samples(self) -> None:
        """Fetch new samples to replace submitted ones."""
        await self._perform_backfill_operation()

    async def _perform_backfill_operation(self) -> None:
        """Perform the backfill operation with appropriate logging."""
        current_count = self.state.queue.total_count
        desired_count = self.state.sample_limit
        needed = max(0, desired_count - current_count)

        # Handle case where queue is already at capacity
        if needed <= 0:
            self.state.update_status("✓ Ready", "green")
            logger.info(f"Queue already at capacity: {current_count}/{desired_count}")
            return

        # Fetch new samples
        self.state.update_status("⚡ Fetching", "yellow")
        logger.info(
            f"Backfilling queue: need {needed} samples "
            f"to reach limit of {desired_count}"
        )

        new_samples_dict = await self.app._fetch_additional_samples(needed)

        if new_samples_dict and len(new_samples_dict) > 0:
            await self._process_new_samples(new_samples_dict)
        else:
            self._handle_no_samples_available(needed)

    async def _process_new_samples(self, new_samples_dict: dict) -> None:
        """Process newly fetched samples and update the queue."""
        new_items = list(new_samples_dict.values())
        self.state.queue.add_items(new_items)

        self.state.update_status("✓ Ready", "green")
        logger.info(f"Successfully added {len(new_items)} new samples to queue")

        # Refresh display if currently viewing an empty state
        if self.state.current_df is None and len(new_items) > 0:
            await self.app.refresh_display()

    def _handle_no_samples_available(self, needed: int) -> None:
        """Handle the case where no new samples are available."""
        self.state.update_status("◯ Empty", "dim")
        logger.warning(f"No new samples available - requested {needed} but got none")
