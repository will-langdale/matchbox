"""Core app tests - simplified to test behavior, not implementation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from matchbox.client.cli.eval.app import EntityResolutionApp, EvaluationQueue
from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.exceptions import MatchboxClientSettingsException


class TestAppInitialization:
    """Test app initialization and configuration."""

    @pytest.fixture
    def test_resolution(self) -> ModelResolutionPath:
        """Create test resolution path."""
        return ModelResolutionPath(
            collection="test_collection", run=1, name="test_resolution"
        )

    def test_app_initializes_with_config(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that app initializes with correct configuration."""
        app = EntityResolutionApp(
            resolution=test_resolution, num_samples=50, user="test_user"
        )

        assert app.resolution == test_resolution
        assert app.sample_limit == 50
        assert app.user_name == "test_user"
        assert isinstance(app.queue, EvaluationQueue)
        assert app.queue.total_count == 0

    def test_compose_creates_ui_structure(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that compose creates the expected UI structure."""
        app = EntityResolutionApp(resolution=test_resolution)

        composed = list(app.compose())

        # Should have Header, main Vertical container, Footer
        assert len(composed) == 3


class TestAuthentication:
    """Test authentication behavior."""

    @pytest.fixture
    def test_resolution(self) -> ModelResolutionPath:
        """Create test resolution path."""
        return ModelResolutionPath(
            collection="test_collection", run=1, name="test_resolution"
        )

    @pytest.mark.asyncio
    async def test_authentication_required(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that authentication requires a user."""
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

            assert app.user_name == "injected_user"
            assert app.user_id == 456
            mock_login.assert_called_once_with(user_name="injected_user")


class TestSampleLoading:
    """Test sample loading behavior."""

    @pytest.fixture
    def test_resolution(self) -> ModelResolutionPath:
        """Create test resolution path."""
        return ModelResolutionPath(
            collection="test_collection", run=1, name="test_resolution"
        )

    @pytest.mark.asyncio
    async def test_load_samples_adds_to_queue(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test loading evaluation samples adds them to queue."""
        app = EntityResolutionApp(resolution=test_resolution)
        app.user_id = 123

        mock_samples = {1: Mock(), 2: Mock(), 3: Mock()}
        app._fetch_additional_samples = AsyncMock(return_value=mock_samples)

        await app.load_samples()

        app._fetch_additional_samples.assert_called_once_with(5)
        assert app.queue.total_count == 3

    @pytest.mark.asyncio
    async def test_fetch_uses_dag(self, test_resolution: ModelResolutionPath) -> None:
        """Test that _fetch_additional_samples uses the loaded DAG."""
        app = EntityResolutionApp(resolution=test_resolution)
        app.user_id = 123
        app.dag = Mock()

        with patch("matchbox.client.cli.eval.app.get_samples") as mock_get_samples:
            mock_get_samples.return_value = {}

            await app._fetch_additional_samples(10)

            mock_get_samples.assert_called_once_with(
                n=10, resolution=test_resolution, user_id=123, dag=app.dag
            )


class TestActions:
    """Test action methods (skip, submit, clear)."""

    @pytest.fixture
    def test_resolution(self) -> ModelResolutionPath:
        """Create test resolution path."""
        return ModelResolutionPath(
            collection="test_collection", run=1, name="test_resolution"
        )

    @pytest.mark.asyncio
    async def test_action_skip_rotates_queue(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that skip action rotates the queue."""
        app = EntityResolutionApp(resolution=test_resolution)

        # Add items to queue
        item1 = Mock(cluster_id=1, assignments={}, display_columns=[1, 2])
        item2 = Mock(cluster_id=2, assignments={}, display_columns=[1, 2])
        app.queue.items.extend([item1, item2])

        # Mock refresh_display since it queries widgets
        app.refresh_display = Mock()

        await app.action_skip()

        # First item should now be at back
        assert app.queue.items[0] == item2
        assert app.queue.items[1] == item1
        app.refresh_display.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_submit_incomplete_shows_warning(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that submitting incomplete entity shows warning."""
        app = EntityResolutionApp(resolution=test_resolution)
        app.user_id = 123

        # Add incomplete item (not painted - assignments don't cover all columns)
        item = Mock(assignments={}, display_columns=[1, 2])
        app.queue.items.append(item)

        # Mock update_status since it queries widgets
        app.update_status = Mock()

        await app.action_submit()

        # Should show incomplete warning
        app.update_status.assert_called_once()
        call_args = app.update_status.call_args[0]
        assert "Incomplete" in call_args[0]

    @pytest.mark.asyncio
    async def test_action_clear_resets_assignments(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that clear action resets assignments."""
        app = EntityResolutionApp(resolution=test_resolution)

        # Add item with assignments
        item = Mock(
            assignments={"a": 1, "b": 2},
            duplicate_groups=[[1], [2]],
            display_columns=[1, 2],
        )
        app.queue.items.append(item)
        app.current_group = "a"

        # Mock query_one since app isn't running
        with patch.object(app, "query_one") as mock_query:
            mock_table = Mock()
            mock_label_left = Mock()
            mock_label_right = Mock()
            mock_query.side_effect = [mock_table, mock_label_left, mock_label_right]

            await app.action_clear()

            assert len(item.assignments) == 0
            assert app.current_group == ""


class TestModals:
    """Test modal screens."""

    @pytest.fixture
    def test_resolution(self) -> ModelResolutionPath:
        """Create test resolution path."""
        return ModelResolutionPath(
            collection="test_collection", run=1, name="test_resolution"
        )

    @pytest.mark.asyncio
    async def test_show_help_modal(self, test_resolution: ModelResolutionPath) -> None:
        """Test that help modal can be shown."""
        app = EntityResolutionApp(resolution=test_resolution)
        app.push_screen = Mock()

        await app.action_show_help()

        app.push_screen.assert_called_once()

    @pytest.mark.asyncio
    async def test_show_no_samples_modal(
        self, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that no samples modal can be shown."""
        app = EntityResolutionApp(resolution=test_resolution)
        app.push_screen = Mock()

        await app.action_show_no_samples()

        app.push_screen.assert_called_once()


class TestEvaluationQueue:
    """Test the inline queue class."""

    def test_queue_initializes_empty(self) -> None:
        """Test queue starts empty."""
        queue = EvaluationQueue()

        assert queue.total_count == 0
        assert queue.current is None

    def test_add_items_increases_count(self) -> None:
        """Test adding items increases count."""
        queue = EvaluationQueue()
        items = [Mock(cluster_id=1), Mock(cluster_id=2)]

        added = queue.add_items(items)

        assert added == 2
        assert queue.total_count == 2

    def test_add_items_prevents_duplicates(self) -> None:
        """Test that duplicate cluster IDs are not added."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        item2 = Mock(cluster_id=1)  # Duplicate

        queue.add_items([item1])
        added = queue.add_items([item2])

        assert added == 0
        assert queue.total_count == 1

    def test_skip_rotates_deque(self) -> None:
        """Test that skip moves current to back."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        item2 = Mock(cluster_id=2)
        queue.items.extend([item1, item2])

        queue.skip_current()

        assert queue.current == item2
        assert queue.items[1] == item1

    def test_remove_current_pops_front(self) -> None:
        """Test that remove_current removes from front."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        item2 = Mock(cluster_id=2)
        queue.items.extend([item1, item2])

        removed = queue.remove_current()

        assert removed == item1
        assert queue.total_count == 1
        assert queue.current == item2
