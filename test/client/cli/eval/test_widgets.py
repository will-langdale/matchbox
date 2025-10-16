"""Unit tests for UI widgets."""

from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest
from rich.table import Table
from rich.text import Text

from matchbox.client.cli.eval.widgets.status import (
    StatusBar,
    StatusBarLeft,
    StatusBarRight,
)
from matchbox.client.cli.eval.widgets.styling import GroupStyler
from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable


class TestComparisonDisplayTable:
    """Test the comparison display table widget."""

    @pytest.fixture
    def mock_current_item(self) -> Mock:
        """Create a mock current queue item."""
        item = Mock()
        item.display_columns = [1, 2, 3]
        item.duplicate_groups = [[1], [2], [3]]

        item.display_dataframe = pl.DataFrame(
            {
                "field_name": ["name", "name", "address", "address"],
                "leaf_id": [1, 2, 1, 3],
                "value": ["Company A", "Company A", "123 Main St", "123 Main Street"],
            }
        )

        return item

    def test_table_initialisation(self, mock_state: Mock) -> None:
        """Test table widget initialisation."""
        table = ComparisonDisplayTable(mock_state)

        assert table.state is mock_state
        mock_state.add_listener.assert_called_once()

    def test_render_no_current_item(self, mock_state: Mock) -> None:
        """Test rendering when no current item exists."""
        mock_state.queue.current = None
        table = ComparisonDisplayTable(mock_state)

        result = table.render()

        assert isinstance(result, Table)
        # Should show loading table

    def test_render_current_item_no_display_dataframe(self, mock_state: Mock) -> None:
        """Test rendering when current item has no display_dataframe."""
        mock_current = Mock()
        mock_current.spec = []  # No display_dataframe attribute
        mock_state.queue.current = mock_current
        table = ComparisonDisplayTable(mock_state)

        result = table.render()

        assert isinstance(result, Table)

    def test_render_compact_view(
        self, mock_state: Mock, mock_current_item: Mock
    ) -> None:
        """Test rendering in compact view mode."""
        mock_state.queue.current = mock_current_item
        table = ComparisonDisplayTable(mock_state)

        result = table.render()

        assert isinstance(result, Table)
        # Should have field name column plus display columns
        assert len(result.columns) == 4  # Field + 3 display columns

    def test_render_with_assignments(
        self, mock_state: Mock, mock_current_item: Mock
    ) -> None:
        """Test rendering with column assignments."""
        mock_state.queue.current = mock_current_item
        mock_state.current_assignments = {0: "a", 2: "b"}
        table = ComparisonDisplayTable(mock_state)

        result = table.render()

        assert isinstance(result, Table)
        # Assignments should affect column headers with colors/symbols

    def test_state_change_triggers_refresh(self, mock_state: Mock) -> None:
        """Test that state changes trigger table refresh."""
        table = ComparisonDisplayTable(mock_state)
        table.refresh = Mock()

        listener = mock_state.add_listener.call_args[0][0]
        listener()

        table.refresh.assert_called_once()


class TestStatusBarLeft:
    """Test the left status bar widget."""

    @pytest.fixture
    def mock_state(self) -> Mock:
        """Create a mock state for testing."""
        state = Mock()
        state.add_listener = Mock()
        state.queue.total_count = 5
        state.queue.current_position = 2
        state.painted_count = 1
        state.has_current_assignments.return_value = True
        state.get_group_counts.return_value = {"a": 3, "b": 2}
        state.current_group_selection = "a"
        return state

    def test_status_bar_left_initialisation(self, mock_state: Mock) -> None:
        """Test left status bar initialisation."""
        status_left = StatusBarLeft(mock_state)

        assert status_left.state is mock_state
        mock_state.add_listener.assert_called_once()

    def test_render_with_data(self, mock_state: Mock) -> None:
        """Test rendering with data."""
        status_left = StatusBarLeft(mock_state)

        result = status_left.render()

        assert isinstance(result, Text)
        # Should contain entity position, painted count, groups info

    def test_render_no_groups(self, mock_state: Mock) -> None:
        """Test rendering when no groups are assigned."""
        mock_state.get_group_counts.return_value = {}
        status_left = StatusBarLeft(mock_state)

        result = status_left.render()

        assert isinstance(result, Text)

    def test_render_no_painted_items(self, mock_state: Mock) -> None:
        """Test rendering when no items are painted."""
        mock_state.painted_count = 0
        mock_state.has_current_assignments.return_value = False
        status_left = StatusBarLeft(mock_state)

        result = status_left.render()

        assert isinstance(result, Text)

    def test_state_change_triggers_refresh(self, mock_state: Mock) -> None:
        """Test that state changes trigger refresh."""
        status_left = StatusBarLeft(mock_state)
        status_left.refresh = Mock()

        listener = mock_state.add_listener.call_args[0][0]
        listener()

        status_left.refresh.assert_called_once()


class TestStatusBarRight:
    """Test the right status bar widget."""

    @pytest.fixture
    def mock_state(self) -> Mock:
        """Create a mock state for testing."""
        state = Mock()
        state.add_listener = Mock()
        state.status_message = ""
        state.status_color = "bright_white"
        return state

    def test_status_bar_right_initialisation(self, mock_state: Mock) -> None:
        """Test right status bar initialisation."""
        status_right = StatusBarRight(mock_state)

        assert status_right.state is mock_state
        mock_state.add_listener.assert_called_once()

    @pytest.mark.parametrize(
        "message,color,should_render",
        [
            ("", "bright_white", True),
            ("✓ Done", "green", True),
            ("✓ Sent", "green", True),
            ("⏳ Loading", "yellow", True),
            ("⚠ Error", "red", True),
            ("◯ Empty", "dim", True),
            ("📊 Got 5", "green", True),
            ("This message is way too long for the status bar", "red", True),
        ],
    )
    def test_status_message_rendering(
        self, mock_state: Mock, message: str, color: str, should_render: bool
    ) -> None:
        """Test rendering of various status messages."""
        mock_state.status_message = message
        mock_state.status_color = color
        status_right = StatusBarRight(mock_state)

        result = status_right.render()

        assert isinstance(result, Text) == should_render

    def test_max_status_length_constant(self) -> None:
        """Test that MAX_STATUS_LENGTH is set correctly."""
        assert StatusBarRight.MAX_STATUS_LENGTH == 12

    def test_state_change_triggers_refresh(self, mock_state: Mock) -> None:
        """Test that state changes trigger refresh."""
        status_right = StatusBarRight(mock_state)
        status_right.refresh = Mock()

        listener = mock_state.add_listener.call_args[0][0]
        listener()

        status_right.refresh.assert_called_once()


class TestStatusBar:
    """Test the status bar container widget."""

    def test_status_bar_initialisation(self, mock_state: Mock) -> None:
        """Test status bar container initialisation."""
        status_bar = StatusBar(mock_state)

        assert status_bar.state is mock_state

    def test_status_bar_compose(self, mock_state: Mock) -> None:
        """Test status bar composition."""
        status_bar = StatusBar(mock_state)

        with patch(
            "matchbox.client.cli.eval.widgets.status.Horizontal"
        ) as mock_horizontal:
            mock_container = MagicMock()
            mock_horizontal.return_value.__enter__.return_value = mock_container
            mock_horizontal.return_value.__exit__.return_value = None

            composed = list(status_bar.compose())

            assert len(composed) == 2


class TestGroupStyler:
    """Test the group styling utility."""

    def teardown_method(self) -> None:
        """Reset GroupStyler state after each test."""
        GroupStyler.reset()

    def test_get_style_consistent(self) -> None:
        """Test that get_style returns consistent results for same group."""
        style1 = GroupStyler.get_style("group_a")
        style2 = GroupStyler.get_style("group_a")

        assert style1 == style2
        assert len(style1) == 2  # (color, symbol)

    def test_get_style_different_groups(self) -> None:
        """Test that different groups get different styles."""
        style_a = GroupStyler.get_style("group_a")
        style_b = GroupStyler.get_style("group_b")

        assert style_a != style_b
        # At least one of color or symbol should be different
        assert style_a[0] != style_b[0] or style_a[1] != style_b[1]

    def test_get_display_text(self) -> None:
        """Test display text formatting."""
        text, _ = GroupStyler.get_display_text("test", 5)

        assert "TEST" in text  # Should be uppercase
        assert "(5)" in text  # Should include count
        assert text.startswith(
            GroupStyler.get_style("test")[1]
        )  # Should start with symbol

    def test_color_cycling(self) -> None:
        """Test that colors cycle when exhausted."""
        num_colors = len(GroupStyler.COLOURS)
        styles = []

        for i in range(num_colors + 2):
            style = GroupStyler.get_style(f"group_{i}")
            styles.append(style)

        colors_used = [style[0] for style in styles]
        unique_colors = set(colors_used)
        assert len(unique_colors) <= num_colors

    def test_symbol_cycling(self) -> None:
        """Test that symbols cycle when exhausted."""
        num_symbols = len(GroupStyler.SYMBOLS)
        styles = []

        for i in range(num_symbols + 2):
            style = GroupStyler.get_style(f"group_{i}")
            styles.append(style)

        symbols_used = [style[1] for style in styles]
        unique_symbols = set(symbols_used)
        assert len(unique_symbols) <= num_symbols

    def test_reset(self) -> None:
        """Test that reset clears all state."""
        GroupStyler.get_style("test1")
        GroupStyler.get_style("test2")

        GroupStyler.reset()

        assert len(GroupStyler._group_styles) == 0
        assert len(GroupStyler._used_colours) == 0
        assert len(GroupStyler._used_symbols) == 0
        assert GroupStyler._colour_index == 0
        assert GroupStyler._symbol_index == 0

    def test_avoid_duplicates_when_possible(self) -> None:
        """Test that duplicates are avoided when colors/symbols are available."""
        styles = []
        for i in range(5):
            style = GroupStyler.get_style(f"group_{i}")
            styles.append(style)

        colors = [style[0] for style in styles]
        symbols = [style[1] for style in styles]

        assert len(set(colors)) == len(colors)
        assert len(set(symbols)) == len(symbols)
