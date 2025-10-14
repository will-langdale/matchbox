"""Unit tests for plot data validation logic."""

from unittest.mock import Mock

import pytest

from matchbox.client.cli.eval.plot.data import (
    can_show_plot,
    refresh_judgements_for_plot,
)


class TestCanShowPlot:
    """Test the can_show_plot function - the core of slash key fix."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        state = Mock()
        state.is_loading_eval_data = False
        state.eval_data_error = None
        state.eval_data = Mock()
        return state

    def test_loading_state_prevents_plot(self, mock_state):
        """Test that loading state prevents plot display."""
        mock_state.is_loading_eval_data = True

        can_show, message = can_show_plot(mock_state)

        assert can_show is False
        assert message == "⏳ Loading"

    def test_error_state_prevents_plot(self, mock_state):
        """Test that error state prevents plot display."""
        mock_state.eval_data_error = "Some error"

        can_show, message = can_show_plot(mock_state)

        assert can_show is False
        assert message == "⚠ Error"

    def test_no_eval_data_prevents_plot(self, mock_state):
        """Test that missing eval data prevents plot display."""
        mock_state.eval_data = None

        can_show, message = can_show_plot(mock_state)

        assert can_show is False
        assert message == "⚠ No data"

    def test_empty_precision_recall_prevents_plot(self, mock_state):
        """Test that empty precision recall data prevents plot display."""
        mock_state.eval_data.precision_recall.return_value = None

        can_show, message = can_show_plot(mock_state)

        assert can_show is False
        assert message == "∅ Sparse"

    def test_insufficient_precision_recall_data_prevents_plot(self, mock_state):
        """Test that insufficient precision recall data prevents plot display."""
        mock_state.eval_data.precision_recall.return_value = [0.5]  # Only 1 point

        can_show, message = can_show_plot(mock_state)

        assert can_show is False
        assert message == "∅ Sparse"

    def test_sufficient_data_allows_plot(self, mock_state):
        """Test that sufficient data allows plot display."""
        mock_state.eval_data.precision_recall.return_value = [0.3, 0.5, 0.8]  # 3 points

        can_show, message = can_show_plot(mock_state)

        assert can_show is True
        assert not message

    def test_priority_order_of_checks(self, mock_state):
        """Test that checks are performed in correct priority order."""
        mock_state.is_loading_eval_data = True
        mock_state.eval_data_error = "Some error"
        mock_state.eval_data = None

        _, message = can_show_plot(mock_state)
        assert message == "⏳ Loading"

        mock_state.is_loading_eval_data = False
        _, message = can_show_plot(mock_state)
        assert message == "⚠ Error"

        mock_state.eval_data_error = None
        _, message = can_show_plot(mock_state)
        assert message == "⚠ No data"

    def test_precision_recall_method_called(self, mock_state):
        """Test that precision_recall method is called when data exists."""
        pr_data = [0.1, 0.5, 0.9]
        mock_state.eval_data.precision_recall.return_value = pr_data

        can_show, _ = can_show_plot(mock_state)

        mock_state.eval_data.precision_recall.assert_called_once()
        assert can_show is True


class TestRefreshJudgementsForPlot:
    """Test judgement refresh functionality."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state with eval_data."""
        state = Mock()
        state.eval_data = Mock()
        state.eval_data.judgements = []
        return state

    def test_successful_refresh_with_judgements(self, mock_state):
        """Test successful judgement refresh with data."""
        mock_state.eval_data.judgements = [Mock(), Mock(), Mock()]  # 3 judgements

        success, message = refresh_judgements_for_plot(mock_state)

        assert success is True
        assert message == "📊 Got 3"
        mock_state.eval_data.refresh_judgements.assert_called_once()

    def test_successful_refresh_no_judgements(self, mock_state):
        """Test successful judgement refresh with no data."""
        mock_state.eval_data.judgements = None

        success, message = refresh_judgements_for_plot(mock_state)

        assert success is True
        assert message == "◯ Empty"
        mock_state.eval_data.refresh_judgements.assert_called_once()

    def test_successful_refresh_empty_judgements(self, mock_state):
        """Test successful judgement refresh with empty list."""
        mock_state.eval_data.judgements = []

        success, message = refresh_judgements_for_plot(mock_state)

        assert success is True
        assert message == "◯ Empty"
        mock_state.eval_data.refresh_judgements.assert_called_once()

    def test_expected_empty_judgement_error(self, mock_state):
        """Test handling of expected 'cannot be empty' error."""
        error = ValueError("judgement dataset cannot be empty")
        mock_state.eval_data.refresh_judgements.side_effect = error

        success, message = refresh_judgements_for_plot(mock_state)

        assert success is True
        assert message == "◯ Empty"

    def test_unexpected_error_handling(self, mock_state):
        """Test handling of unexpected errors."""
        error = ConnectionError("Network error")
        mock_state.eval_data.refresh_judgements.side_effect = error

        success, message = refresh_judgements_for_plot(mock_state)

        assert success is False
        assert message == "⚠ Error"

    def test_other_expected_judgement_errors(self, mock_state):
        """Test handling of other judgement-related errors treated as empty."""
        judgement_errors = [
            "Judgement cannot be empty for evaluation",
            "Cannot compute metrics with empty judgement data",
            "judgements dataset must not be empty",
        ]

        for error_msg in judgement_errors:
            mock_state.eval_data.refresh_judgements.side_effect = ValueError(error_msg)

            success, message = refresh_judgements_for_plot(mock_state)

            assert success is True
            assert message == "◯ Empty"


class TestSlashKeyRegressionScenarios:
    """Focused regression tests for the original slash key modal bug."""

    @pytest.mark.parametrize(
        "scenario_name,is_loading,error,has_eval_data,pr_data,expected_can_show,expected_msg",
        [
            ("fresh_app_no_data", False, None, False, None, False, "⚠ No data"),
            ("loading_eval_data", True, None, True, None, False, "⏳ Loading"),
            (
                "eval_data_load_error",
                False,
                "Model not found",
                False,
                None,
                False,
                "⚠ Error",
            ),
            (
                "eval_data_loaded_no_judgements",
                False,
                None,
                True,
                [],
                False,
                "∅ Sparse",
            ),
            (
                "eval_data_loaded_single_point",
                False,
                None,
                True,
                [0.5],
                False,
                "∅ Sparse",
            ),
            (
                "eval_data_ready_for_plotting",
                False,
                None,
                True,
                [0.2, 0.5, 0.8],
                True,
                "",
            ),
        ],
    )
    def test_all_slash_key_scenarios(
        self,
        scenario_name,
        is_loading,
        error,
        has_eval_data,
        pr_data,
        expected_can_show,
        expected_msg,
    ):
        """Test all scenarios that could cause the slash key bug."""
        state = Mock()
        state.is_loading_eval_data = is_loading
        state.eval_data_error = error
        state.eval_data = Mock() if has_eval_data else None

        if state.eval_data and pr_data is not None:
            state.eval_data.precision_recall.return_value = pr_data

        can_show, message = can_show_plot(state)

        assert can_show == expected_can_show, f"Scenario '{scenario_name}' failed"
        assert message == expected_msg, f"Scenario '{scenario_name}' failed"

    @pytest.mark.parametrize(
        "scenario_name,is_loading,error,has_eval_data,pr_data,expected_msg",
        [
            ("fresh_app_no_data", False, None, False, None, "⚠ No data"),
            ("loading_eval_data", True, None, True, None, "⏳ Loading"),
            ("eval_data_load_error", False, "Model not found", False, None, "⚠ Error"),
            ("eval_data_loaded_no_judgements", False, None, True, [], "∅ Sparse"),
            ("eval_data_loaded_single_point", False, None, True, [0.5], "∅ Sparse"),
        ],
    )
    def test_slash_key_never_shows_modal_when_not_ready(
        self, scenario_name, is_loading, error, has_eval_data, pr_data, expected_msg
    ):
        """Critical test: slash key should NEVER show modal when plot is not ready."""
        state = Mock()
        state.is_loading_eval_data = is_loading
        state.eval_data_error = error
        state.eval_data = Mock() if has_eval_data else None

        if state.eval_data and pr_data is not None:
            state.eval_data.precision_recall.return_value = pr_data

        can_show, message = can_show_plot(state)

        assert (
            can_show is False
        ), f"CRITICAL: Scenario '{scenario_name}' should prevent modal display"
        assert (
            message
        ), f"CRITICAL: Scenario '{scenario_name}' should return status message"
