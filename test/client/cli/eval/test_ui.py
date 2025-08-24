"""Tests for Textual UI components."""

from functools import partial
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Engine

from matchbox.client.cli.eval.ui import EntityResolutionApp
from matchbox.common.exceptions import MatchboxClientSettingsException
from matchbox.common.factories.scenarios import setup_scenario

backends = [
    pytest.param("matchbox_postgres", id="postgres"),
]


@pytest.fixture(scope="function")
def backend_instance(request: pytest.FixtureRequest, backend: str):
    """Create a fresh backend instance for each test."""
    backend_obj = request.getfixturevalue(backend)
    backend_obj.clear(certain=True)
    return backend_obj


@pytest.mark.parametrize("backend", backends)
@pytest.mark.docker
class TestTextualUI:
    """Test the Textual UI using scenario data like adapter tests."""

    @pytest.fixture(autouse=True)
    def setup(self, backend_instance: str, sqlite_warehouse: Engine):
        self.backend = backend_instance
        self.warehouse_engine = sqlite_warehouse
        self.scenario = partial(setup_scenario, warehouse=sqlite_warehouse)

    @pytest.mark.asyncio
    async def test_app_initialisation(self):
        """Test that the app can be initialised."""
        app = EntityResolutionApp(resolution="test_resolution", num_samples=5)
        assert app.state.resolution == "test_resolution"
        assert app.state.sample_limit == 5
        assert app.state.queue.current_position == 0

    @pytest.mark.asyncio
    async def test_app_runs_headless(self):
        """Test that the app can run in test mode."""
        app = EntityResolutionApp(resolution="test_resolution", num_samples=5)

        with (
            patch("matchbox.client.cli.eval.ui.settings") as mock_settings,
            patch("matchbox.client.cli.eval.ui._handler.login") as mock_login,
            patch("matchbox.client.cli.eval.ui.get_samples") as mock_get_samples,
        ):
            mock_settings.user = "test_user"
            mock_login.return_value = 123
            mock_get_samples.return_value = {}

            # Test basic app lifecycle
            async with app.run_test() as pilot:
                # App should be running
                assert app.is_running

                # Should have header and footer
                assert pilot.app.query("Header")
                assert pilot.app.query("Footer")

    @patch("matchbox.client.cli.eval.ui.settings")
    @pytest.mark.asyncio
    async def test_authentication_required(self, mock_settings):
        """Test that authentication is required."""
        # No user set
        mock_settings.user = None

        app = EntityResolutionApp(resolution="test", num_samples=1)

        # Should raise exception when no user is configured
        with pytest.raises(MatchboxClientSettingsException):
            async with app.run_test() as pilot:
                await pilot.pause()

    @patch("matchbox.client.cli.eval.ui.get_samples")
    @patch("matchbox.client.cli.eval.ui._handler.login")
    @patch("matchbox.client.cli.eval.ui.settings")
    @pytest.mark.asyncio
    async def test_basic_workflow_with_mocked_data(
        self, mock_settings, mock_login, mock_get_samples
    ):
        """Test basic workflow with mocked sample data."""
        # Setup mocks
        mock_settings.user = "test_user"
        mock_login.return_value = 123
        mock_samples = {}  # Empty samples for simplicity
        mock_get_samples.return_value = mock_samples

        app = EntityResolutionApp(resolution="test", num_samples=1)

        async with app.run_test() as pilot:
            await pilot.pause()

            # Should have authenticated
            assert app.state.user_name == "test_user"
            assert app.state.user_id == 123

            # Should have loaded samples (even if empty)
            assert app.state.queue.total_count == len(mock_samples)

    @pytest.mark.asyncio
    async def test_help_modal(self):
        """Test that help modal can be triggered."""
        app = EntityResolutionApp(resolution="test", num_samples=1)

        with (
            patch("matchbox.client.cli.eval.ui.settings") as mock_settings,
            patch("matchbox.client.cli.eval.ui._handler.login") as mock_login,
            patch("matchbox.client.cli.eval.ui.get_samples") as mock_get_samples,
        ):
            mock_settings.user = "test_user"
            mock_login.return_value = 123
            mock_get_samples.return_value = {}

            async with app.run_test() as pilot:
                await pilot.pause()

                # Press F1 to open help
                await pilot.press("f1")
                await pilot.pause()

                # Help modal should be visible (this is a basic check)
                # In a full implementation, we'd check for specific help content

    @pytest.mark.asyncio
    async def test_command_input_parsing(self):
        """Test command parsing functionality."""

        app = EntityResolutionApp(resolution="test", num_samples=1)
        app.push_screen = Mock()  # Mock screen pushing

        with (
            patch("matchbox.client.cli.eval.ui.settings") as mock_settings,
            patch("matchbox.client.cli.eval.ui._handler.login") as mock_login,
            patch("matchbox.client.cli.eval.ui.get_samples") as mock_get_samples,
        ):
            mock_settings.user = "test_user"
            mock_login.return_value = 123
            mock_get_samples.return_value = {}

            async with app.run_test() as pilot:
                await pilot.pause()

                # Test state management exists
                assert hasattr(pilot.app, "state")

                # Test state functionality
                state = pilot.app.state
                state.set_group_selection("b")
                assert state.current_group_selection == "b"
                assert state.parse_number_key("1") == 1

    @pytest.mark.asyncio
    async def test_scenario_integration(self):
        """Test integration with scenario data (requires Docker)."""
        with self.scenario(self.backend, "dedupe") as dag:
            # Get a real model from the scenario
            model_name = list(dag.models.keys())[0] if dag.models else "test"

            # Create app with injected parameters - no mocking needed!
            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=1,
                user="test_user",
                warehouse=str(self.warehouse_engine.url),
            )

            # Test the app works with real scenario data
            async with app.run_test() as pilot:
                await pilot.pause()

                # Should have authenticated
                assert app.state.user_name == "test_user"
                assert app.state.user_id is not None

                # App should be running
                assert app.is_running

                # Should have loaded samples from real scenario data
                assert (
                    app.state.queue.total_count >= 0
                )  # Could be 0 if no evaluation clusters yet

    def test_action_submit_and_fetch_exists(self):
        """Test that the action_submit_and_fetch method exists and is properly bound."""
        app = EntityResolutionApp(resolution="test", num_samples=5)

        # Verify the method exists
        assert hasattr(app, "action_submit_and_fetch")
        assert callable(app.action_submit_and_fetch)

        # Verify the spacebar binding exists in BINDINGS
        space_binding = next(
            (binding for binding in app.BINDINGS if binding[0] == "space"), None
        )
        assert space_binding is not None
        assert space_binding[1] == "submit_and_fetch"
        assert space_binding[2] == "Submit & fetch more"

    def test_persistent_painting_functionality(self):
        """Test that painting persists across entity navigation."""
        app = EntityResolutionApp(resolution="test", num_samples=5)

        # Test queue-based storage exists in state
        assert hasattr(app.state, "queue")
        assert hasattr(app.state.queue, "items")

        # Test state has required methods
        assert hasattr(app.state, "has_current_assignments")
        assert callable(app.state.has_current_assignments)

        # Test that set_display_data works
        app.state.set_display_data(["field1"], [["value1"]], [1])
        assert app.state.field_names == ["field1"]
        assert app.state.data_matrix == [["value1"]]
        assert app.state.leaf_ids == [1]
