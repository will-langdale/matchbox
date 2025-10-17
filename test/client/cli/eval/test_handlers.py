"""Unit tests for input handlers and actions."""

from unittest.mock import AsyncMock, Mock, PropertyMock

import pytest

from matchbox.client.cli.eval.handlers import EvaluationHandlers


class TestEvaluationHandlers:
    """Test the EvaluationHandlers class."""

    @pytest.fixture
    def mock_app(self) -> Mock:
        """Create a mock app for testing."""
        app = Mock()
        app.state = Mock()
        app.push_screen = Mock()
        app.refresh_display = AsyncMock()
        app.action_quit = AsyncMock()
        app._fetch_additional_samples = AsyncMock(return_value={})
        app.state.add_queue_items = Mock(return_value=0)
        app.state.mark_submitted = Mock()

        mock_current = Mock()
        mock_current.display_columns = []
        app.state.queue.current = None

        return app

    @pytest.fixture
    def handlers(self, mock_app: Mock) -> EvaluationHandlers:
        """Create handlers instance with mock app."""
        return EvaluationHandlers(mock_app)

    @pytest.fixture
    def mock_event(self) -> Mock:
        """Create a mock key event."""
        event = Mock()
        event.prevent_default = Mock()
        return event

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", ["left", "right", "enter", "space"])
    async def test_navigation_keys_not_handled(
        self, handlers: EvaluationHandlers, mock_event: Mock, key: str
    ) -> None:
        """Test that navigation keys are passed through to bindings."""
        mock_event.key = key
        await handlers.handle_key_input(mock_event)
        mock_event.prevent_default.assert_not_called()

    @pytest.mark.asyncio
    async def test_escape_clears_group_selection(
        self, handlers: EvaluationHandlers, mock_event: Mock
    ) -> None:
        """Test that escape key clears group selection."""
        mock_event.key = "escape"

        await handlers.handle_key_input(mock_event)

        handlers.state.clear_group_selection.assert_called_once()
        mock_event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("letter", ["a", "b", "z", "A", "B", "Z"])
    async def test_letter_keys_set_group_selection(
        self, handlers: EvaluationHandlers, mock_event: Mock, letter: str
    ) -> None:
        """Test that letter keys set group selection."""
        mock_event.key = letter
        await handlers.handle_key_input(mock_event)
        handlers.state.set_group_selection.assert_called_once_with(letter)
        mock_event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", ["1", "!", "@"])
    async def test_non_alpha_keys_ignored(
        self, handlers: EvaluationHandlers, mock_event: Mock, key: str
    ) -> None:
        """Test that non-alphabetic keys don't set group selection."""
        mock_event.key = key
        await handlers.handle_key_input(mock_event)
        handlers.state.set_group_selection.assert_not_called()

    @pytest.mark.asyncio
    async def test_number_keys_assign_columns(
        self, handlers: EvaluationHandlers, mock_event: Mock
    ) -> None:
        """Test that number keys assign columns to current group."""
        handlers.state.current_group_selection = "a"
        handlers.state.parse_number_key.return_value = 3

        mock_current = Mock()
        mock_current.display_columns = [1, 2, 3, 4, 5]
        handlers.state.queue.current = mock_current

        mock_event.key = "3"

        await handlers.handle_key_input(mock_event)

        handlers.state.parse_number_key.assert_called_once_with("3")
        handlers.state.assign_column_to_group.assert_called_once_with(3, "a")
        mock_event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    async def test_number_keys_no_group_selected(
        self, handlers: EvaluationHandlers, mock_event: Mock
    ) -> None:
        """Test that number keys do nothing when no group is selected."""
        handlers.state.current_group_selection = ""
        handlers.state.parse_number_key.return_value = 3

        mock_event.key = "3"

        await handlers.handle_key_input(mock_event)

        handlers.state.assign_column_to_group.assert_not_called()
        mock_event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    async def test_next_entity_multiple_items(
        self, handlers: EvaluationHandlers
    ) -> None:
        """Test moving to next entity when multiple items exist."""
        handlers.state.queue.total_count = 3

        await handlers.action_next_entity()

        handlers.state.queue.move_next.assert_called_once()
        handlers.app.refresh_display.assert_called_once()

    @pytest.mark.asyncio
    async def test_next_entity_single_item_submits_and_quits(
        self, handlers: EvaluationHandlers
    ) -> None:
        """Test moving to next when only one item submits and quits."""
        handlers.state.queue.total_count = 1
        handlers.action_submit_and_fetch = AsyncMock()

        await handlers.action_next_entity()

        handlers.state.queue.move_next.assert_not_called()
        handlers.action_submit_and_fetch.assert_called_once()
        handlers.app.action_quit.assert_called_once()

    @pytest.mark.asyncio
    async def test_previous_entity_multiple_items(
        self, handlers: EvaluationHandlers
    ) -> None:
        """Test moving to previous entity when multiple items exist."""
        handlers.state.queue.total_count = 3

        await handlers.action_previous_entity()

        handlers.state.queue.move_previous.assert_called_once()
        handlers.app.refresh_display.assert_called_once()

    @pytest.mark.asyncio
    async def test_previous_entity_single_item_does_nothing(
        self, handlers: EvaluationHandlers
    ) -> None:
        """Test moving to previous when only one item does nothing."""
        handlers.state.queue.total_count = 1

        await handlers.action_previous_entity()

        handlers.state.queue.move_previous.assert_not_called()
        handlers.app.refresh_display.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_assignments(self, handlers: EvaluationHandlers) -> None:
        """Test clearing assignments and group selection."""
        await handlers.action_clear_assignments()

        handlers.state.clear_current_assignments.assert_called_once()
        handlers.state.clear_group_selection.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_and_fetch_multiple_painted(
        self, handlers: EvaluationHandlers, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Submit multiple painted items and backfill queue."""
        item_one = Mock()
        item_one.cluster_id = 1
        item_one.to_judgement.return_value = Mock()
        item_two = Mock()
        item_two.cluster_id = 2
        item_two.to_judgement.return_value = Mock()

        handlers.state.painted_items = [item_one, item_two]
        handlers.state.user_id = 42
        handlers.state.mark_submitted = Mock()
        handlers.state.queue.total_count = 5
        handlers.state.update_status.reset_mock()

        send_mock = Mock()
        monkeypatch.setattr(
            "matchbox.client.cli.eval.handlers._handler.send_eval_judgement",
            send_mock,
        )

        handlers._backfill_samples = AsyncMock()

        await handlers.action_submit_and_fetch()

        assert send_mock.call_count == 2
        send_mock.assert_any_call(judgement=item_one.to_judgement.return_value)
        send_mock.assert_any_call(judgement=item_two.to_judgement.return_value)
        handlers.state.mark_submitted.assert_called_once_with({1, 2})
        handlers._backfill_samples.assert_called_once()
        # Final status call should indicate success
        assert any(
            call.args[0].startswith("✓ Sent")
            for call in handlers.state.update_status.call_args_list
        )

    @pytest.mark.asyncio
    async def test_show_help(self, handlers: EvaluationHandlers) -> None:
        """Test showing help modal."""
        await handlers.action_show_help()

        handlers.app.push_screen.assert_called_once()
        # Verify it's a HelpModal (by checking the call)
        call_args = handlers.app.push_screen.call_args[0][0]
        assert call_args.__class__.__name__ == "HelpModal"

    @pytest.mark.asyncio
    async def test_backfill_samples_success(self, handlers: EvaluationHandlers) -> None:
        """Test successful sample backfilling."""
        # Simulate queue count increasing from 80 to 100 as items are added
        # Each add_queue_items call adds 2 items, so: 80, 82, 84, ..., 100
        # Need extra 100 at end for final check after loop completes
        queue_counts = [80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 100]
        type(handlers.state.queue).total_count = PropertyMock(side_effect=queue_counts)
        handlers.state.sample_limit = 100
        handlers.app._fetch_additional_samples = AsyncMock(
            return_value={"1": Mock(), "2": Mock()}
        )
        handlers.state.current_df = Mock()  # Not empty state
        handlers.state.add_queue_items.return_value = 2

        await handlers._backfill_samples()

        # Should fetch samples 10 times to go from 80 to 100
        assert handlers.app._fetch_additional_samples.call_count == 10
        assert handlers.state.add_queue_items.call_count == 10

    @pytest.mark.asyncio
    async def test_backfill_samples_already_at_capacity(
        self, handlers: EvaluationHandlers
    ) -> None:
        """Test backfill when queue is already at capacity."""
        handlers.state.queue.total_count = 100
        handlers.state.sample_limit = 100

        await handlers._backfill_samples()

        handlers.app._fetch_additional_samples.assert_not_called()
        handlers.state.update_status.assert_called_with("✓ Ready", "green")

    @pytest.mark.asyncio
    async def test_backfill_samples_no_samples_available(
        self, handlers: EvaluationHandlers
    ) -> None:
        """Test backfill when no new samples are available."""
        handlers.state.queue.total_count = 80
        handlers.state.sample_limit = 100
        handlers.app._fetch_additional_samples = AsyncMock(return_value=None)

        await handlers._backfill_samples()

        handlers.state.update_status.assert_called_with("◯ Empty", "dim")
