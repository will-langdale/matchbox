"""Minimal widget tests - test that widgets render, not implementation details."""

import polars as pl
import pytest
from rich.table import Table
from rich.text import Text

from matchbox.client.cli.eval.widgets.status import StatusBar
from matchbox.client.cli.eval.widgets.styling import get_group_style
from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable
from matchbox.client.eval import EvaluationItem


class TestComparisonDisplayTable:
    """Test table widget renders correctly."""

    @pytest.fixture
    def mock_item(self) -> EvaluationItem:
        """Create a mock evaluation item."""
        df = pl.DataFrame({"leaf": [1, 2, 3]})

        item = EvaluationItem(
            cluster_id=1,
            dataframe=df,
            display_data={
                "name": ["Company A", "Company B", ""],
                "address": ["", "", "123 Main St"],
            },
            duplicate_groups=[[1], [2], [3]],
            display_columns=[1, 2, 3],
            assignments={},
        )
        return item

    def test_table_renders(self, mock_item: EvaluationItem) -> None:
        """Test that table can render."""
        table = ComparisonDisplayTable()
        table.load_comparison(mock_item)

        result = table.render()

        assert isinstance(result, Table)

    def test_table_renders_with_assignments(self, mock_item: EvaluationItem) -> None:
        """Test that table renders with column assignments."""
        mock_item.assignments = {0: "a", 2: "b"}

        table = ComparisonDisplayTable()
        table.load_comparison(mock_item)

        result = table.render()

        assert isinstance(result, Table)
        assert len(result.columns) == 4  # Field + 3 display columns

    def test_table_handles_no_item(self) -> None:
        """Test that table handles no current item."""
        table = ComparisonDisplayTable()

        result = table.render()

        assert isinstance(result, Table)


class TestStatusBar:
    """Test status bar widget renders correctly."""

    def test_status_bar_renders(self) -> None:
        """Test that status bar can render."""
        status_bar = StatusBar()
        status_bar.queue_position = 1
        status_bar.queue_total = 5
        status_bar.group_counts = {"a": 3, "b": 2}
        status_bar.current_group = "a"

        result = status_bar.render()

        assert isinstance(result, Table)

    def test_status_bar_renders_empty_state(self) -> None:
        """Test that status bar renders empty state."""
        status_bar = StatusBar()
        status_bar.queue_position = 0
        status_bar.queue_total = 0

        result = status_bar.render()

        assert isinstance(result, Table)

    def test_status_message_display(self) -> None:
        """Test that status messages display correctly."""
        status_bar = StatusBar()
        status_bar.status_message = "✓ Sent"
        status_bar.status_color = "green"

        result = status_bar._render_right()

        assert isinstance(result, Text)
        assert "✓ Sent" in str(result)


class TestGroupStyler:
    """Test group styling utilities."""

    def test_static_lookup_consistent(self) -> None:
        """Test that group styling is deterministic."""
        style1 = get_group_style("a")
        style2 = get_group_style("a")

        assert style1 == style2
        assert len(style1) == 2  # (symbol, color)

    def test_different_groups_different_styles(self) -> None:
        """Test that different groups get different styles."""
        style_a = get_group_style("a")
        style_b = get_group_style("b")

        # At least one should be different (symbol or color)
        assert style_a != style_b
