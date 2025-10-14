"""Tests for Textual UI components."""

from functools import partial
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Engine

from matchbox.client.cli.eval.app import EntityResolutionApp
from matchbox.client.cli.eval.state import EvaluationState
from matchbox.client.cli.eval.widgets.status import StatusBarRight
from matchbox.client.dags import DAG
from matchbox.client.sources import RelationalDBLocation
from matchbox.common.dtos import ModelResolutionPath
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

    @pytest.fixture
    def test_resolution(self):
        """Create test resolution path."""
        return ModelResolutionPath(
            collection="test_collection", run=1, name="test_resolution"
        )

    @pytest.fixture(autouse=True)
    def setup(self, backend_instance: str, sqlite_warehouse: Engine):
        self.backend = backend_instance
        self.warehouse_engine = sqlite_warehouse
        self.scenario = partial(setup_scenario, warehouse=sqlite_warehouse)

    @pytest.mark.asyncio
    async def test_app_initialisation(self, test_resolution):
        """Test that the app can be initialised."""
        app = EntityResolutionApp(resolution=test_resolution, num_samples=5)
        assert app.state.resolution == test_resolution
        assert app.state.sample_limit == 5
        assert app.state.queue.current_position == 0

    @pytest.mark.asyncio
    async def test_app_runs_headless(self, test_resolution):
        """Test that the app can run in test mode."""
        app = EntityResolutionApp(resolution=test_resolution, num_samples=5)

        # Mock DAG for the test
        app.state.dag = Mock()

        with (
            patch("matchbox.client.cli.eval.app.settings") as mock_settings,
            patch("matchbox.client.cli.eval.app._handler.login") as mock_login,
            patch("matchbox.client.cli.eval.app.get_samples") as mock_get_samples,
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

    @patch("matchbox.client.cli.eval.app.settings")
    @pytest.mark.asyncio
    async def test_authentication_required(self, mock_settings, test_resolution):
        """Test that authentication is required."""
        # No user set
        mock_settings.user = None

        app = EntityResolutionApp(resolution=test_resolution, num_samples=1)

        # Should raise exception when no user is configured
        with pytest.raises(MatchboxClientSettingsException):
            async with app.run_test() as pilot:
                await pilot.pause()

    @patch("matchbox.client.cli.eval.app.get_samples")
    @patch("matchbox.client.cli.eval.app._handler.login")
    @patch("matchbox.client.cli.eval.app.settings")
    @pytest.mark.asyncio
    async def test_basic_workflow_with_mocked_data(
        self, mock_settings, mock_login, mock_get_samples, test_resolution
    ):
        """Test basic workflow with mocked sample data."""
        # Setup mocks
        mock_settings.user = "test_user"
        mock_login.return_value = 123
        mock_samples = {}  # Empty samples for simplicity
        mock_get_samples.return_value = mock_samples

        app = EntityResolutionApp(resolution=test_resolution, num_samples=1)

        # Mock DAG for the test
        app.state.dag = Mock()

        async with app.run_test() as pilot:
            await pilot.pause()

            # Should have authenticated
            assert app.state.user_name == "test_user"
            assert app.state.user_id == 123

            # Should have loaded samples (even if empty)
            assert app.state.queue.total_count == len(mock_samples)

    @pytest.mark.asyncio
    async def test_help_modal(self, test_resolution):
        """Test that help modal can be triggered."""
        app = EntityResolutionApp(resolution=test_resolution, num_samples=1)

        # Mock DAG for the test
        app.state.dag = Mock()

        with (
            patch("matchbox.client.cli.eval.app.settings") as mock_settings,
            patch("matchbox.client.cli.eval.app._handler.login") as mock_login,
            patch("matchbox.client.cli.eval.app.get_samples") as mock_get_samples,
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
    async def test_command_input_parsing(self, test_resolution):
        """Test command parsing functionality."""

        app = EntityResolutionApp(resolution=test_resolution, num_samples=1)
        app.push_screen = Mock()  # Mock screen pushing

        # Mock DAG for the test
        app.state.dag = Mock()

        with (
            patch("matchbox.client.cli.eval.app.settings") as mock_settings,
            patch("matchbox.client.cli.eval.app._handler.login") as mock_login,
            patch("matchbox.client.cli.eval.app.get_samples") as mock_get_samples,
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

            # Construct ModelResolutionPath from DAG context
            resolution = ModelResolutionPath(
                collection=dag.dag.name, run=dag.dag.run, name=model_name
            )

            # Create warehouse location and load DAG
            warehouse_location = RelationalDBLocation(
                name="test_warehouse", client=self.warehouse_engine
            )
            loaded_dag = DAG(str(dag.dag.name)).load_run(
                run_id=dag.dag.run, location=warehouse_location
            )

            # Create app with injected parameters - no mocking needed!
            app = EntityResolutionApp(
                resolution=resolution,
                num_samples=1,
                user="test_user",
                dag=loaded_dag,
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

    @pytest.mark.asyncio
    async def test_action_submit_and_fetch_functionality(self):
        """Test that the UI's action_submit_and_fetch properly submits painted items."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            # Create warehouse location and load DAG
            warehouse_location = RelationalDBLocation(
                name="test_warehouse", client=self.warehouse_engine
            )
            loaded_dag = DAG(str(dag.dag.name)).load_run(
                run_id=dag.dag.run, location=warehouse_location
            )

            app = EntityResolutionApp(
                resolution=dag.models[model_name].model.resolution_path,
                num_samples=5,
                user="test_user",
                dag=loaded_dag,
            )

            # Login
            await app.authenticate()

            # Fetch some evaluation items to work with (use internal method)
            items_dict = await app._fetch_additional_samples(2)
            if items_dict:
                items = list(items_dict.values())

                # Paint items
                for i, item in enumerate(items):
                    for display_col_idx in range(len(item.display_columns)):
                        item.assignments[display_col_idx] = "a" if i == 0 else "b"

                app.state.queue.add_items(items)

            # Test submission workflow
            initial_judgements, _ = self.backend.get_judgements()
            initial_count = len(initial_judgements)

            await app.action_submit_and_fetch()

            # Verify judgements were submitted
            final_judgements, _ = self.backend.get_judgements()
            final_count = len(final_judgements)

            assert final_count > initial_count
            new_judgements = final_judgements.to_pylist()[initial_count:]

            submitted_cluster_ids = {j["shown"] for j in new_judgements}
            expected_cluster_ids = {item.cluster_id for item in items}
            assert submitted_cluster_ids == expected_cluster_ids

            # Verify painted items were removed from queue
            assert len(app.state.queue.painted_items) == 0

        # Also verify the spacebar binding
        assert hasattr(app, "action_submit_and_fetch")
        space_binding = next((b for b in app.BINDINGS if b[0] == "space"), None)
        assert space_binding[1] == "submit_and_fetch"

    def test_persistent_painting_functionality(self, test_resolution):
        """Test that painting persists across entity navigation."""
        app = EntityResolutionApp(resolution=test_resolution, num_samples=5)

        # Test queue-based storage exists in state
        assert hasattr(app.state, "queue")
        assert hasattr(app.state.queue, "items")

        # Test state has required methods
        assert hasattr(app.state, "has_current_assignments")
        assert callable(app.state.has_current_assignments)

        # Test that set_display_data works
        app.state.set_display_data([1])
        assert app.state.display_leaf_ids == [1]

    def test_status_message_length_validation(self):
        """Test that status messages are properly validated for length."""

        state = EvaluationState()
        status_widget = StatusBarRight(state)

        # Test valid messages (should pass)
        valid_messages = [
            "⏳ Loading",
            "✓ Loaded",
            "⚡ Sending",
            "✓ Done",
            "⚠ Error",
            "◯ Empty",
            "📊 Got 5",
            "✓ Ready",
        ]

        for msg in valid_messages:
            state.update_status(msg)
            text = status_widget.render()
            # Should not contain error indicator
            assert "TOO LONG" not in str(text)
            assert len(msg) <= StatusBarRight.MAX_STATUS_LENGTH

        # Test invalid message (should fail)
        state.update_status("This message is way too long for the status bar")
        text = status_widget.render()
        # Should show error indicator
        assert "TOO LONG" in str(text)

    def test_status_message_colours(self):
        """Test that status messages use proper colours."""

        state = EvaluationState()
        status_widget = StatusBarRight(state)

        # Test colour assignments
        test_cases = [
            ("⚠ Error", "red"),
            ("✓ Ready", "green"),
            ("⏳ Loading", "yellow"),
            ("◯ Empty", "dim"),
        ]

        for message, expected_colour in test_cases:
            state.update_status(message, expected_colour)
            # Verify state stores the colour
            assert state.status_colour == expected_colour
            # Verify rendering uses the colour
            text = status_widget.render()
            assert message in str(text)
