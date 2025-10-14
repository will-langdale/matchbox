"""Shared fixtures for CLI eval tests."""

from unittest.mock import Mock, patch

import polars as pl
import pytest

from matchbox.client.cli.eval.utils import EvalData


@pytest.fixture
def mock_current_item():
    """Create a standard mock current queue item for testing."""
    item = Mock()
    item.cluster_id = "test_cluster"
    item.dataframe = Mock()
    item.assignments = {0: "a", 2: "b"}
    item.display_columns = [1, 2, 3, 4]
    item.duplicate_groups = [[1], [2], [3], [4]]

    # Standard display dataframe
    item.display_dataframe = pl.DataFrame(
        {
            "field_name": ["name", "name", "address", "address"],
            "leaf_id": [1, 2, 1, 3],
            "value": ["Company A", "Company A", "123 Main St", "123 Main Street"],
            "source_name": ["crn", "duns", "crn", "cdms"],
        }
    )

    return item


@pytest.fixture
def mock_state():
    """Create a standard mock state for testing."""
    state = Mock()
    state.add_listener = Mock()
    state.compact_view_mode = True
    state.current_assignments = {}
    state.queue.current = None
    return state


@pytest.fixture
def evaldata_factory():
    """Factory fixture for creating EvalData instances with test data."""

    def _create_evaldata(
        judgements, expansion, model_root_leaf, thresholds=None, probabilities=None
    ):
        if thresholds is None:
            thresholds = [1.0]

        with patch(
            "matchbox.client.cli.eval.utils._handler.download_eval_data"
        ) as mock:
            mock.return_value = (judgements, expansion)

            return EvalData(
                root_leaf=model_root_leaf,
                thresholds=thresholds,
                probabilities=probabilities,
            )

    return _create_evaldata


@pytest.fixture
def mock_eval_dependencies():
    """Mock common eval dependencies for app testing."""
    with (
        patch("matchbox.client.cli.eval.app.settings") as mock_settings,
        patch("matchbox.client.cli.eval.app._handler.login") as mock_login,
        patch("matchbox.client.cli.eval.app.get_samples") as mock_get_samples,
    ):
        mock_settings.user = "test_user"
        mock_login.return_value = 123
        mock_get_samples.return_value = {}
        yield {
            "settings": mock_settings,
            "login": mock_login,
            "get_samples": mock_get_samples,
        }
