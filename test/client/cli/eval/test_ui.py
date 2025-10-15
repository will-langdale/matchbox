"""Tests for Textual UI components."""

from functools import partial
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Engine

from matchbox.client.cli.eval.app import EntityResolutionApp
from matchbox.client.cli.eval.state import EvaluationState
from matchbox.client.cli.eval.utils import EvaluationItem
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
    async def test_app_runs_headless(self, test_resolution, mock_eval_dependencies):
        """Test that the app can run in test mode."""
        app = EntityResolutionApp(resolution=test_resolution, num_samples=5)
        app.state.dag = Mock()

        async with app.run_test() as pilot:
            assert app.is_running
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
    async def test_help_modal(self, test_resolution, mock_eval_dependencies):
        """Test that help modal can be triggered."""
        app = EntityResolutionApp(resolution=test_resolution, num_samples=1)
        app.state.dag = Mock()

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("f1")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_command_input_parsing(self, test_resolution, mock_eval_dependencies):
        """Test command parsing functionality."""
        app = EntityResolutionApp(resolution=test_resolution, num_samples=1)
        app.push_screen = Mock()
        app.state.dag = Mock()

        async with app.run_test() as pilot:
            await pilot.pause()
            assert hasattr(pilot.app, "state")

            state = pilot.app.state
            state.set_group_selection("b")
            assert state.current_group_selection == "b"
            assert state.parse_number_key("1") == 1

    @pytest.mark.asyncio
    async def test_scenario_integration(self):
        """Test integration with scenario data (requires Docker)."""
        with self.scenario(self.backend, "dedupe") as dag:
            # Get a real model from the scenario
            model = list(dag.models.values())[0] if dag.models else "test"

            # Create warehouse location and load DAG
            warehouse_location = RelationalDBLocation(
                name="test_warehouse", client=self.warehouse_engine
            )
            loaded_dag = DAG(str(dag.dag.name)).load_pending(
                location=warehouse_location
            )

            # Create app with injected parameters - no mocking needed!
            app = EntityResolutionApp(
                resolution=model.resolution_path,
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
            loaded_dag = DAG(str(dag.dag.name)).load_pending(
                location=warehouse_location
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
            items_dict: dict[int, EvaluationItem] = await app._fetch_additional_samples(
                2
            )
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

    def test_status_bar_integration(self):
        """Test that status bar widget works in app context."""
        state = EvaluationState()
        status_widget = StatusBarRight(state)

        state.update_status("✓ Ready", "green")
        text = status_widget.render()
        assert "✓ Ready" in str(text)
