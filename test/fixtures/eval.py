"""Shared fixtures for CLI eval tests."""

from collections.abc import Iterator
from unittest.mock import Mock, patch

import polars as pl
import pytest


@pytest.fixture
def mock_current_item() -> Mock:
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
def mock_state() -> Mock:
    """Create a standard mock state for testing."""
    state = Mock()
    state.add_listener = Mock()
    state.compact_view_mode = True
    state.current_assignments = {}
    state.queue.current = None
    return state


@pytest.fixture
def mock_eval_dependencies() -> Iterator[dict[str, Mock]]:
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
