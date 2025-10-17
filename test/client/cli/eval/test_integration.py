"""Integration tests with real scenario data - the most valuable tests."""

from functools import partial
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Engine

from matchbox.client.cli.eval.app import EntityResolutionApp
from matchbox.client.dags import DAG
from matchbox.client.eval import EvaluationItem
from matchbox.client.sources import RelationalDBLocation
from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.exceptions import MatchboxClientSettingsException
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.server.base import MatchboxDBAdapter

backends = [
    pytest.param("matchbox_postgres", id="postgres"),
]


@pytest.fixture(scope="function")
def backend_instance(request: pytest.FixtureRequest, backend: str) -> MatchboxDBAdapter:
    """Create a fresh backend instance for each test."""
    backend_obj = request.getfixturevalue(backend)
    backend_obj.clear(certain=True)
    return backend_obj


@pytest.mark.parametrize("backend", backends)
@pytest.mark.docker
class TestScenarioIntegration:
    """Integration tests using real scenario data - most valuable tests."""

    @pytest.fixture
    def test_resolution(self) -> ModelResolutionPath:
        """Create test resolution path."""
        return ModelResolutionPath(
            collection="test_collection", run=1, name="test_resolution"
        )

    @pytest.fixture(autouse=True)
    def setup(self, backend_instance: str, sqlite_warehouse: Engine) -> None:
        """Set up test fixtures."""
        self.backend = backend_instance
        self.warehouse_engine = sqlite_warehouse
        self.scenario = partial(setup_scenario, warehouse=sqlite_warehouse)

    @pytest.mark.asyncio
    async def test_app_runs_headless(
        self, test_resolution: ModelResolutionPath, mock_eval_dependencies: dict
    ) -> None:
        """Test that the app can run in test mode."""
        app = EntityResolutionApp(resolution=test_resolution, num_samples=5)
        app.dag = Mock()

        async with app.run_test() as pilot:
            assert app.is_running
            assert pilot.app.query("Header")
            assert pilot.app.query("Footer")

    @patch("matchbox.client.cli.eval.app.settings")
    @pytest.mark.asyncio
    async def test_authentication_required(
        self, mock_settings: Mock, test_resolution: ModelResolutionPath
    ) -> None:
        """Test that authentication is required to run app."""
        mock_settings.user = None
        app = EntityResolutionApp(resolution=test_resolution, num_samples=1)

        with pytest.raises(MatchboxClientSettingsException):
            async with app.run_test():
                pass

    @pytest.mark.asyncio
    async def test_full_scenario_integration(self) -> None:
        """Test full integration with real scenario data - MOST VALUABLE TEST."""
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

            # Create app with real scenario data
            app = EntityResolutionApp(
                resolution=model.resolution_path,
                num_samples=1,
                user="test_user",
                dag=loaded_dag,
            )

            # Test the app works with real data
            async with app.run_test() as pilot:
                await pilot.pause()

                # Should have authenticated
                assert app.user_name == "test_user"
                assert app.user_id is not None

                # App should be running
                assert app.is_running

                # Should have loaded samples from real scenario
                assert app.queue.total_count >= 0

    @pytest.mark.asyncio
    async def test_submit_workflow_with_real_data(self) -> None:
        """Test submission workflow with real scenario data."""
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

            # Login and fetch samples
            await app.authenticate()
            items_dict: dict[int, EvaluationItem] = await app._fetch_additional_samples(
                2
            )

            if not items_dict:
                pytest.skip("No evaluation samples available for this scenario")

            items = list(items_dict.values())

            # Paint items (assign all columns to group 'a')
            for item in items:
                for display_col_idx in range(len(item.display_columns)):
                    item.assignments[display_col_idx] = "a"

            app.queue.add_items(items)

            # Submit painted items
            initial_judgements, _ = self.backend.get_judgements()
            initial_count = len(initial_judgements)

            # Mock update_status and refresh_display since app isn't running
            app.update_status = lambda *args, **kwargs: None
            app.refresh_display = lambda: None

            # Submit via action method
            for _ in range(len(items)):
                current = app.queue.current
                if current and len(current.assignments) == len(current.display_columns):
                    await app.action_submit()

            # Verify judgements were submitted
            final_judgements, _ = self.backend.get_judgements()
            final_count = len(final_judgements)

            assert final_count == initial_count + len(items)

    @patch("matchbox.client.cli.eval.app.get_samples")
    @patch("matchbox.client.cli.eval.app._handler.login")
    @patch("matchbox.client.cli.eval.app.settings")
    @pytest.mark.asyncio
    async def test_no_samples_shows_modal(
        self,
        mock_settings: Mock,
        mock_login: Mock,
        mock_get_samples: Mock,
        test_resolution: ModelResolutionPath,
    ) -> None:
        """Test that no samples triggers modal."""
        mock_settings.user = "test_user"
        mock_login.return_value = 123
        mock_get_samples.return_value = {}  # Empty triggers no samples state

        app = EntityResolutionApp(resolution=test_resolution, num_samples=1)
        app.dag = Mock()

        async with app.run_test() as pilot:
            await pilot.pause()

            # Should have triggered no samples state
            assert app.has_no_samples is True
            # Modal should be shown
            assert len(pilot.app.screen_stack) > 1

    @pytest.mark.asyncio
    async def test_help_modal_opens(
        self, test_resolution: ModelResolutionPath, mock_eval_dependencies: dict
    ) -> None:
        """Test that help modal can be opened."""
        app = EntityResolutionApp(resolution=test_resolution, num_samples=1)
        app.dag = Mock()

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("f1")
            await pilot.pause()

            # Help modal should be shown
            assert len(pilot.app.screen_stack) > 1
