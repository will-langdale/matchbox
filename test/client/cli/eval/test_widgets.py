"""Minimal widget tests - verify rendering only."""

import polars as pl
import pytest
from rich.table import Table

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
                "name": ["Company A", "Company B", "Company C"],
                "address": ["123 Main", "456 Oak", "789 Pine"],
            },
            duplicate_groups=[[1], [2], [3]],
            display_columns=[1, 2, 3],
            assignments={},
        )
        return item

    def test_table_renders_with_assignments(self, mock_item: EvaluationItem) -> None:
        """Test that table renders with column assignments."""
        mock_item.assignments = {0: "a", 2: "b"}

        table = ComparisonDisplayTable()
        table.load_comparison(
            mock_item,
            col_start=0,
            col_end=2,
            row_start=0,
            row_end=1,
        )

        result = table.render()

        assert isinstance(result, Table)
        assert len(result.columns) == 3  # Field + 2 display columns

    def test_table_handles_no_item(self) -> None:
        """Test that table handles no current item."""
        table = ComparisonDisplayTable()

        result = table.render()

        assert isinstance(result, Table)
