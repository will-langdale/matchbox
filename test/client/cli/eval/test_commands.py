"""Tests for keyboard shortcut functionality."""

from matchbox.client.cli.eval.ui import GroupStyler, KeyboardShortcutHandler


class TestKeyboardShortcuts:
    """Test keyboard shortcut functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = KeyboardShortcutHandler()

    def test_single_letter_group(self):
        """Test single letter group selection."""
        self.handler.on_key_down("a")
        assert self.handler.get_current_group() == "a"

        self.handler.on_key_up("a")
        assert self.handler.get_current_group() == ""

    def test_multi_letter_group_sequence(self):
        """Test building multi-letter groups in sequence."""
        # Build A+S+D sequence
        self.handler.on_key_down("a")
        assert self.handler.get_current_group() == "a"

        self.handler.on_key_down("s")
        assert self.handler.get_current_group() == "as"

        self.handler.on_key_down("d")
        assert self.handler.get_current_group() == "asd"

        # Release middle letter
        self.handler.on_key_up("s")
        assert self.handler.get_current_group() == "ad"

        # Release all
        self.handler.on_key_up("a")
        self.handler.on_key_up("d")
        assert self.handler.get_current_group() == ""

    def test_duplicate_key_handling(self):
        """Test that duplicate keys don't get added twice."""
        self.handler.on_key_down("a")
        self.handler.on_key_down("a")  # Duplicate
        assert self.handler.get_current_group() == "a"

        self.handler.on_key_down("s")
        self.handler.on_key_down("s")  # Duplicate
        assert self.handler.get_current_group() == "as"

    def test_case_insensitive_keys(self):
        """Test that uppercase keys are converted to lowercase."""
        self.handler.on_key_down("A")
        assert self.handler.get_current_group() == "a"

        self.handler.on_key_down("S")
        assert self.handler.get_current_group() == "as"

    def test_number_key_parsing(self):
        """Test number key to row number conversion."""
        assert self.handler.parse_number_key("1") == 1
        assert self.handler.parse_number_key("5") == 5
        assert self.handler.parse_number_key("9") == 9
        assert self.handler.parse_number_key("0") == 10  # 0 maps to row 10
        assert self.handler.parse_number_key("a") is None  # Non-number returns None

    def test_flexible_letter_combinations(self):
        """Test various letter combinations work."""
        # QWERTY sequence
        for key in "qwerty":
            self.handler.on_key_down(key)
        assert self.handler.get_current_group() == "qwerty"

        # Reset
        for key in "qwerty":
            self.handler.on_key_up(key)

        # Home row sequence
        for key in "asdf":
            self.handler.on_key_down(key)
        assert self.handler.get_current_group() == "asdf"

    def test_non_alphabetic_keys_ignored(self):
        """Test that non-alphabetic keys are ignored."""
        self.handler.on_key_down("1")  # Number
        self.handler.on_key_down("!")  # Special character
        self.handler.on_key_down("space")  # Word key
        assert self.handler.get_current_group() == ""


class TestGroupStyler:
    """Test the GroupStyler class for consistent color/symbol assignment."""

    def test_style_consistency(self):
        """Test that same group name always gets same color/symbol."""
        test_groups = ["qwerty", "asdf", "xyz", "hello", "world"]

        for group in test_groups:
            style1 = GroupStyler.get_style(group)
            style2 = GroupStyler.get_style(group)
            assert style1 == style2, f"Inconsistent style for group: {group}"

    def test_different_groups_different_styles(self):
        """Test that different group names get different styles (usually)."""
        # While hash collisions are possible, these specific groups should differ
        style_a = GroupStyler.get_style("a")
        style_qwerty = GroupStyler.get_style("qwerty")

        # At least one component (color or symbol) should be different
        assert style_a != style_qwerty

    def test_display_text_format(self):
        """Test display text formatting."""
        text, color = GroupStyler.get_display_text("qwerty", 5)

        assert "QWERTY" in text
        assert "(5)" in text
        assert color in GroupStyler.COLORS

    def test_color_symbol_within_bounds(self):
        """Test that generated colors and symbols are within defined ranges."""
        for group in ["a", "qwerty", "xyz", "hello", "test123"]:
            color, symbol = GroupStyler.get_style(group)

            assert color in GroupStyler.COLORS
            assert symbol in GroupStyler.SYMBOLS
