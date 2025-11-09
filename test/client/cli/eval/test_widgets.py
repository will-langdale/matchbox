"""Minimal widget tests - verify construction only."""

from rich.text import Text

from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable


class TestComparisonDisplayTable:
    """Test table widget construction."""

    def test_table_initializes(self) -> None:
        """Test that table can be created."""
        table = ComparisonDisplayTable()
        assert table is not None
        assert table.current_item is None
        assert table.current_assignments == {}
        assert table.current_group == ""
        assert table.zebra_stripes is True
        assert table.cursor_type == "none"
        assert table.show_cursor is False

    def test_build_header_without_assignment(self) -> None:
        """Test that _build_header creates correct header without assignment."""
        table = ComparisonDisplayTable()

        # Build header for position 1 with no duplicates
        header = table._build_header(1, [1], None)
        assert isinstance(header, Text)
        assert header.plain == "1"

        # Build header for position 2 with duplicates
        header = table._build_header(2, [1, 2, 3], None)
        assert header.plain == "2 (×3)"

    def test_build_header_with_assignment(self) -> None:
        """Test that _build_header creates correct header with assignment."""
        table = ComparisonDisplayTable()

        # Build header with assignment 'a' (◇ - hollow diamond)
        header = table._build_header(1, [1], "a")
        assert header.plain.startswith("◇")  # Symbol for group 'a'
        assert "1" in header.plain
        # Should have exactly one symbol
        assert header.plain.count("◇") == 1

        # Build header with assignment 'b' (⬤ - filled circle)
        header = table._build_header(2, [1, 2], "b")
        assert header.plain.startswith("⬤")  # Symbol for group 'b'
        assert "2 (×2)" in header.plain
        assert header.plain.count("⬤") == 1

    def test_build_header_different_assignments_no_accumulation(self) -> None:
        """Test headers with different assignments don't accumulate symbols."""
        table = ComparisonDisplayTable()

        # Build header with assignment 'a' (◇ - hollow diamond)
        header1 = table._build_header(1, [1], "a")
        assert header1.plain.count("◇") == 1
        assert "⬤" not in header1.plain

        # Build header for same position with assignment 'b' (⬤ - filled circle)
        header2 = table._build_header(1, [1], "b")
        assert header2.plain.count("⬤") == 1
        assert "◇" not in header2.plain  # No old symbol

        # Build header for same position with assignment 'c' (◻ - hollow square)
        header3 = table._build_header(1, [1], "c")
        assert header3.plain.startswith("◻")  # Symbol for group 'c'
        assert "◇" not in header3.plain  # No old symbols
        assert "⬤" not in header3.plain

    def test_get_column_positions(self) -> None:
        """Test that _get_column_positions returns correct positions."""
        table = ComparisonDisplayTable()

        # Without current_item, should return empty list
        assert table._get_column_positions() == []
