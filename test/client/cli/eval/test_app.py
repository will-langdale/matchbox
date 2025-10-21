"""Integration tests with real scenario data - comprehensive behaviour testing."""

from functools import partial
from unittest.mock import Mock

import pytest
from sqlalchemy import Engine

from matchbox.client.cli.eval.app import EntityResolutionApp, EvaluationQueue
from matchbox.client.dags import DAG
from matchbox.common.dtos import ModelResolutionPath
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


class TestEvaluationQueue:
    """Unit tests for EvaluationQueue - no app or mocking needed."""

    def test_queue_initialises_empty(self) -> None:
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
        assert queue.current is items[0]

    def test_add_items_prevents_duplicates(self) -> None:
        """Test that duplicate cluster IDs are not added."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        item2 = Mock(cluster_id=1)  # Duplicate

        queue.add_items([item1])
        added = queue.add_items([item2])

        assert added == 0
        assert queue.total_count == 1

    def test_add_items_handles_empty_list(self) -> None:
        """Test that adding empty list returns 0."""
        queue = EvaluationQueue()

        added = queue.add_items([])

        assert added == 0
        assert queue.total_count == 0

    def test_skip_rotates_deque(self) -> None:
        """Test that skip moves current to back."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        item2 = Mock(cluster_id=2)
        queue.items.extend([item1, item2])

        queue.skip_current()

        assert queue.current is item2
        assert queue.items[1] is item1

    def test_skip_with_single_item_does_nothing(self) -> None:
        """Test that skip with one item doesn't rotate."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        queue.items.append(item1)

        queue.skip_current()

        assert queue.current is item1
        assert queue.total_count == 1

    def test_remove_current_pops_front(self) -> None:
        """Test that remove_current removes from front."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        item2 = Mock(cluster_id=2)
        queue.items.extend([item1, item2])

        removed = queue.remove_current()

        assert removed is item1
        assert queue.total_count == 1
        assert queue.current is item2

    def test_remove_current_on_empty_returns_none(self) -> None:
        """Test that remove_current on empty queue returns None."""
        queue = EvaluationQueue()

        removed = queue.remove_current()

        assert removed is None
        assert queue.total_count == 0


@pytest.mark.parametrize("backend", backends)
@pytest.mark.docker
class TestScenarioIntegration:
    """Integration tests using real scenario data with backend."""

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
    async def test_app_runs_with_real_scenario(self) -> None:
        """Test that app runs successfully with real scenario data."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = (
                DAG(str(dag.dag.name)).load_pending().set_client(self.warehouse_engine)
            )

            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=1,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Basic checks
                assert app.is_running
                assert app.user_name == "test_user"
                assert app.user_id is not None
                assert pilot.app.query("Footer")

    @pytest.mark.asyncio
    async def test_sample_loading_with_real_data(self) -> None:
        """Test that samples load from real scenario."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = (
                DAG(str(dag.dag.name)).load_pending().set_client(self.warehouse_engine)
            )

            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=5,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Samples should be loaded
                assert app.queue.total_count >= 0

                if app.queue.total_count > 0:
                    # Should have a current item
                    assert app.queue.current is not None
                    assert app.queue.current.cluster_id is not None

    @pytest.mark.asyncio
    async def test_keyboard_workflow_letter_then_digit(self) -> None:
        """Test the typical keyboard workflow."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = (
                DAG(str(dag.dag.name)).load_pending().set_client(self.warehouse_engine)
            )

            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=1,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Press 'a' to select group
                await pilot.press("a")
                await pilot.pause()
                assert app.current_group == "a"

                # Press '1' to assign first column
                await pilot.press("1")
                await pilot.pause()

                current = app.queue.current
                assert current is not None
                assert 0 in current.assignments
                assert current.assignments[0] == "a"

    @pytest.mark.asyncio
    async def test_clear_action_resets_assignments(self) -> None:
        """Test that clear action resets all assignments."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = (
                DAG(str(dag.dag.name)).load_pending().set_client(self.warehouse_engine)
            )

            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=1,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Make some assignments
                await pilot.press("a")
                await pilot.press("1")
                await pilot.press("b")
                await pilot.press("2")
                await pilot.pause()

                current = app.queue.current
                assert current is not None
                assert len(current.assignments) > 0

                # Clear them
                await pilot.press("escape")
                await pilot.pause()

                assert len(current.assignments) == 0
                assert app.current_group == ""

    @pytest.mark.asyncio
    async def test_skip_workflow(self) -> None:
        """Test skipping moves item to back of queue."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = (
                DAG(str(dag.dag.name)).load_pending().set_client(self.warehouse_engine)
            )

            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=3,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                first_item = app.queue.current

                # Skip current item
                await pilot.press("right")
                await pilot.pause()

                # Should have moved to next item
                assert app.queue.current is not first_item
                # First item should be at the back
                assert app.queue.items[-1] is first_item

    @pytest.mark.asyncio
    async def test_submit_workflow_incomplete_shows_warning(self) -> None:
        """Test that submitting incomplete assignment shows warning."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = (
                DAG(str(dag.dag.name)).load_pending().set_client(self.warehouse_engine)
            )

            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=1,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                initial_count = app.queue.total_count

                # Try to submit without completing assignments
                await pilot.press("space")
                await pilot.pause()

                # Should still have same item (not submitted)
                assert app.queue.total_count == initial_count
                # Should show incomplete warning in status
                status_message = app.status[0].lower()
                assert "incomplete" in status_message

    @pytest.mark.asyncio
    async def test_submit_workflow_complete_sends_judgement(self) -> None:
        """Test that submitting complete assignment sends judgement."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = (
                DAG(str(dag.dag.name)).load_pending().set_client(self.warehouse_engine)
            )

            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=2,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                current = app.queue.current
                assert current is not None

                # Complete all assignments
                num_columns = len(current.display_columns)
                for i in range(num_columns):
                    await pilot.press("a")
                    await pilot.press(str((i % 9) + 1))  # Cycle through 1-9
                    await pilot.pause()

                initial_judgements, _ = self.backend.get_judgements()
                initial_count = len(initial_judgements)

                # Submit
                await pilot.press("space")
                await pilot.pause()

                # Verify judgement was sent
                final_judgements, _ = self.backend.get_judgements()
                final_count = len(final_judgements)

                assert final_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_help_modal_opens_and_closes(self) -> None:
        """Test that help modal can be opened and closed."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = (
                DAG(str(dag.dag.name)).load_pending().set_client(self.warehouse_engine)
            )

            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=1,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                initial_stack_size = len(pilot.app.screen_stack)

                # Open help modal
                await pilot.press("f1")
                await pilot.pause()

                # Modal should be shown
                assert len(pilot.app.screen_stack) > initial_stack_size

    @pytest.mark.asyncio
    async def test_status_updates_reactively(self) -> None:
        """Test that status updates when assignments change."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = (
                DAG(str(dag.dag.name)).load_pending().set_client(self.warehouse_engine)
            )

            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=1,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Status should be a tuple
                assert isinstance(app.status, tuple)
                assert len(app.status) == 2

                # Make an assignment
                await pilot.press("a")
                await pilot.press("1")
                await pilot.pause()

                # Status should still be a valid tuple
                assert isinstance(app.status, tuple)
                assert len(app.status) == 2
