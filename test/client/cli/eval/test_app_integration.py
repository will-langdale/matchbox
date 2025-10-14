"""Integration tests for the main EntityResolutionApp."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from matchbox.client.cli.eval.app import EntityResolutionApp
from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.exceptions import MatchboxClientSettingsException


class TestEntityResolutionAppIntegration:
    """Integration tests for the main app."""

    @pytest.fixture
    def test_resolution(self):
        """Create test resolution path."""
        return ModelResolutionPath(
            collection="test_collection", run=1, name="test_resolution"
        )

    @pytest.fixture
    def app(self, test_resolution):
        """Create app instance for testing."""
        return EntityResolutionApp(resolution=test_resolution, num_samples=5)

    def test_app_initialisation(self, app, test_resolution):
        """Test that app initialises with correct state."""
        assert app.state.resolution == test_resolution
        assert app.state.sample_limit == 5
        assert app.handlers.app is app
        assert app.handlers.state is app.state

    @pytest.mark.asyncio
    async def test_app_runs_in_test_mode(self, test_resolution):
        """Test that the app can run in test mode without errors."""
        app = EntityResolutionApp(resolution=test_resolution, num_samples=5)

        # Mock DAG for the test
        app.state.dag = Mock()

        with (
            patch("matchbox.client.cli.eval.app.settings") as mock_settings,
            patch("matchbox.client.cli.eval.app._handler.login") as mock_login,
            patch("matchbox.client.cli.eval.app.get_samples") as mock_get_samples,
            patch(
                "matchbox.client.cli.eval.app.EvalData.from_resolution"
            ) as mock_eval_data,
        ):
            mock_settings.user = "test_user"
            mock_login.return_value = 123
            mock_get_samples.return_value = {}
            mock_eval_data.return_value = Mock()

            # Should not raise any exceptions during mount
            await app.on_mount()

            assert app.state.user_name == "test_user"
            assert app.state.user_id == 123

    @pytest.mark.asyncio
    async def test_authentication_required(self, test_resolution):
        """Test that authentication is required."""
        app = EntityResolutionApp(resolution=test_resolution)

        with patch("matchbox.client.cli.eval.app.settings") as mock_settings:
            mock_settings.user = None

            with pytest.raises(MatchboxClientSettingsException):
                await app.authenticate()

    @pytest.mark.asyncio
    async def test_authentication_with_injected_user(self, test_resolution):
        """Test authentication with user injected via constructor."""
        app = EntityResolutionApp(resolution=test_resolution, user="injected_user")

        with patch("matchbox.client.cli.eval.app._handler.login") as mock_login:
            mock_login.return_value = 456

            await app.authenticate()

            assert app.state.user_name == "injected_user"
            assert app.state.user_id == 456

    @pytest.mark.asyncio
    async def test_eval_data_loading_success(self, test_resolution):
        """Test successful eval data loading."""
        app = EntityResolutionApp(resolution=test_resolution)

        mock_eval_data = Mock()
        with patch(
            "matchbox.client.cli.eval.app.EvalData.from_resolution"
        ) as mock_from_resolution:
            mock_from_resolution.return_value = mock_eval_data

            await app.load_eval_data()

            assert app.state.eval_data is mock_eval_data
            assert app.state.is_loading_eval_data is False
            assert app.state.eval_data_error is None

    @pytest.mark.asyncio
    async def test_eval_data_loading_error(self):
        """Test eval data loading error handling."""
        error_resolution = ModelResolutionPath(
            collection="test_collection", run=1, name="nonexistent_resolution"
        )
        app = EntityResolutionApp(resolution=error_resolution)

        with patch(
            "matchbox.client.cli.eval.app.EvalData.from_resolution"
        ) as mock_from_resolution:
            mock_from_resolution.side_effect = ValueError("Model not found")

            await app.load_eval_data()

            assert app.state.eval_data is None
            assert app.state.is_loading_eval_data is False
            assert "not found" in app.state.eval_data_error.lower()

    @pytest.mark.asyncio
    async def test_sample_loading(self, test_resolution):
        """Test loading evaluation samples."""
        app = EntityResolutionApp(resolution=test_resolution)
        app.state.user_id = 123

        mock_samples = {1: Mock(), 2: Mock()}
        app._fetch_additional_samples = AsyncMock(return_value=mock_samples)

        await app.load_samples()

        app._fetch_additional_samples.assert_called_once_with(
            100
        )  # Default sample limit
        assert app.state.queue.total_count == 2

    @pytest.mark.asyncio
    async def test_refresh_display_with_current_item(self, test_resolution):
        """Test refresh display with current queue item."""
        app = EntityResolutionApp(resolution=test_resolution)

        mock_item = Mock()
        mock_item.display_columns = [1, 2, 3]
        app.state.queue.items.append(mock_item)

        await app.refresh_display()

        # Should set display data to current item's display columns
        assert app.state.display_leaf_ids == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_refresh_display_no_current_item(self, test_resolution):
        """Test refresh display with no current queue item."""
        app = EntityResolutionApp(resolution=test_resolution)

        await app.refresh_display()

        # Should clear display data
        assert app.state.display_leaf_ids == []

    @pytest.mark.asyncio
    async def test_key_delegation_to_handlers(self, test_resolution):
        """Test that key events are delegated to handlers."""
        app = EntityResolutionApp(resolution=test_resolution)
        app.handlers.handle_key_input = AsyncMock()

        mock_event = Mock()
        await app.on_key(mock_event)

        app.handlers.handle_key_input.assert_called_once_with(mock_event)

    @pytest.mark.asyncio
    async def test_action_delegation_to_handlers(self, test_resolution):
        """Test that actions are delegated to handlers."""
        app = EntityResolutionApp(resolution=test_resolution)

        # Mock all handler methods
        app.handlers.action_next_entity = AsyncMock()
        app.handlers.action_previous_entity = AsyncMock()
        app.handlers.action_clear_assignments = AsyncMock()
        app.handlers.action_toggle_view_mode = AsyncMock()
        app.handlers.action_show_help = AsyncMock()
        app.handlers.action_submit_and_fetch = AsyncMock()

        # Test each action
        await app.action_next_entity()
        app.handlers.action_next_entity.assert_called_once()

        await app.action_previous_entity()
        app.handlers.action_previous_entity.assert_called_once()

        await app.action_clear_assignments()
        app.handlers.action_clear_assignments.assert_called_once()

        await app.action_toggle_view_mode()
        app.handlers.action_toggle_view_mode.assert_called_once()

        await app.action_show_help()
        app.handlers.action_show_help.assert_called_once()

        await app.action_submit_and_fetch()
        app.handlers.action_submit_and_fetch.assert_called_once()

    def test_compose_creates_expected_structure(self, test_resolution):
        """Test that compose creates the expected UI structure."""
        app = EntityResolutionApp(resolution=test_resolution)

        composed = list(app.compose())

        # Should have Header, main Vertical container, Footer
        assert len(composed) == 3

    @pytest.mark.asyncio
    async def test_fetch_additional_samples_with_dag(self, app):
        """Test that _fetch_additional_samples uses the loaded DAG."""
        app.state.user_id = 123

        # Create mock DAG
        mock_dag = Mock()
        app.state.dag = mock_dag

        with patch("matchbox.client.cli.eval.app.get_samples") as mock_get_samples:
            mock_get_samples.return_value = {}

            await app._fetch_additional_samples(10)

            # Verify get_samples called with DAG
            mock_get_samples.assert_called_once_with(
                n=10,
                resolution=app.state.resolution,
                user_id=app.state.user_id,
                dag=mock_dag,
            )

    def test_error_message_creation(self, test_resolution):
        """Test creation of user-friendly error messages."""
        app = EntityResolutionApp(resolution=test_resolution)

        # Test various error types
        not_found_error = ValueError("Model 'test' not found in database")
        msg = app._create_eval_data_error_message(not_found_error)
        assert "not found" in msg.lower()

        empty_error = ValueError("Empty dataset for model")
        msg = app._create_eval_data_error_message(empty_error)
        assert "no data available" in msg.lower()

        generic_error = ConnectionError("Network timeout")
        msg = app._create_eval_data_error_message(generic_error)
        assert "ConnectionError" in msg
