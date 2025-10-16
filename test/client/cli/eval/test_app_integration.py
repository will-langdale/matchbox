"""Integration tests for the main EntityResolutionApp."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from matchbox.client.cli.eval.app import EntityResolutionApp
from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.exceptions import MatchboxClientSettingsException


class TestEntityResolutionAppIntegration:
    """Integration tests for the main app."""

    @pytest.fixture
    def test_resolution(self) -> ModelResolutionPath:
        """Create test resolution path."""
        return ModelResolutionPath(
            collection="test_collection", run=1, name="test_resolution"
        )

    @pytest.fixture
    def app(self, test_resolution: ModelResolutionPath) -> EntityResolutionApp:
        """Create app instance for testing."""
        return EntityResolutionApp(resolution=test_resolution, num_samples=5)

    def test_app_initialisation(
        self, app: EntityResolutionApp, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that app initialises with correct state."""
        assert app.state.resolution == test_resolution
        assert app.state.sample_limit == 5
        assert app.handlers.app is app
        assert app.handlers.state is app.state

    @pytest.mark.asyncio
    async def test_authentication_required(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that authentication is required."""
        app = EntityResolutionApp(resolution=test_resolution)

        with patch("matchbox.client.cli.eval.app.settings") as mock_settings:
            mock_settings.user = None

            with pytest.raises(MatchboxClientSettingsException):
                await app.authenticate()

    @pytest.mark.asyncio
    async def test_authentication_with_injected_user(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test authentication with user injected via constructor."""
        app = EntityResolutionApp(resolution=test_resolution, user="injected_user")

        with patch("matchbox.client.cli.eval.app._handler.login") as mock_login:
            mock_login.return_value = 456

            await app.authenticate()

            assert app.state.user_name == "injected_user"
            assert app.state.user_id == 456

    @pytest.mark.asyncio
    async def test_sample_loading(self, test_resolution: ModelResolutionPath) -> None:
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
    async def test_refresh_display_with_current_item(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test refresh display with current queue item."""
        app = EntityResolutionApp(resolution=test_resolution)

        mock_item = Mock()
        mock_item.display_columns = [1, 2, 3]
        app.state.queue.items.append(mock_item)

        await app.refresh_display()

        # Should set display data to current item's display columns
        assert app.state.display_leaf_ids == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_refresh_display_no_current_item(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test refresh display with no current queue item."""
        app = EntityResolutionApp(resolution=test_resolution)

        await app.refresh_display()

        # Should clear display data
        assert app.state.display_leaf_ids == []

    @pytest.mark.asyncio
    async def test_key_delegation_to_handlers(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that key events are delegated to handlers."""
        app = EntityResolutionApp(resolution=test_resolution)
        app.handlers.handle_key_input = AsyncMock()

        mock_event = Mock()
        await app.on_key(mock_event)

        app.handlers.handle_key_input.assert_called_once_with(mock_event)

    @pytest.mark.asyncio
    async def test_action_delegation_to_handlers(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that actions are delegated to handlers."""
        app = EntityResolutionApp(resolution=test_resolution)

        # Mock all handler methods
        app.handlers.action_next_entity = AsyncMock()
        app.handlers.action_previous_entity = AsyncMock()
        app.handlers.action_clear_assignments = AsyncMock()
        app.handlers.action_show_help = AsyncMock()
        app.handlers.action_submit_and_fetch = AsyncMock()

        # Test each action
        await app.action_next_entity()
        app.handlers.action_next_entity.assert_called_once()

        await app.action_previous_entity()
        app.handlers.action_previous_entity.assert_called_once()

        await app.action_clear_assignments()
        app.handlers.action_clear_assignments.assert_called_once()

        await app.action_show_help()
        app.handlers.action_show_help.assert_called_once()

        await app.action_submit_and_fetch()
        app.handlers.action_submit_and_fetch.assert_called_once()

    def test_compose_creates_expected_structure(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that compose creates the expected UI structure."""
        app = EntityResolutionApp(resolution=test_resolution)

        composed = list(app.compose())

        # Should have Header, main Vertical container, Footer
        assert len(composed) == 3

    @pytest.mark.asyncio
    async def test_fetch_additional_samples_with_dag(
        self, app: EntityResolutionApp
    ) -> None:
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
