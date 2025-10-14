"""Tests for keyboard shortcut functionality."""

import polars as pl
import pytest

from matchbox.client.cli.eval.state import EvaluationState
from matchbox.client.cli.eval.utils import create_evaluation_item
from matchbox.common.factories.sources import (
    source_from_tuple,
)


class TestKeyboardShortcuts:
    """Test keyboard shortcut functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.state = EvaluationState()

    def test_single_letter_group(self):
        """Test single letter group selection."""
        self.state.set_group_selection("a")
        assert self.state.current_group_selection == "a"

        self.state.clear_group_selection()
        assert not self.state.current_group_selection

    def test_group_switching(self):
        """Test switching between different groups."""
        # Set group A
        self.state.set_group_selection("a")
        assert self.state.current_group_selection == "a"

        # Switch to group S
        self.state.set_group_selection("s")
        assert self.state.current_group_selection == "s"

        # Switch to group D
        self.state.set_group_selection("d")
        assert self.state.current_group_selection == "d"

        # Clear group
        self.state.clear_group_selection()
        assert not self.state.current_group_selection

    def test_set_group_overrides(self):
        """Test that setting a group overrides the previous group."""
        self.state.set_group_selection("a")
        assert self.state.current_group_selection == "a"

        # Setting a new group replaces the old one
        self.state.set_group_selection("b")
        assert self.state.current_group_selection == "b"

        # Setting same group again should still work
        self.state.set_group_selection("b")
        assert self.state.current_group_selection == "b"

    def test_case_insensitive_keys(self):
        """Test that uppercase keys are converted to lowercase."""
        self.state.set_group_selection("A")
        assert self.state.current_group_selection == "a"

        self.state.set_group_selection("S")
        assert self.state.current_group_selection == "s"

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("1", 1),
            ("5", 5),
            ("9", 9),
            ("0", 10),
            ("a", None),
        ],
    )
    def test_number_key_parsing(self, key, expected):
        """Test number key to column number conversion."""
        assert self.state.parse_number_key(key) == expected

    def test_various_single_letter_groups(self):
        """Test various single letter groups work."""
        # Test different letters
        test_letters = ["q", "w", "e", "r", "t", "y", "a", "s", "d", "f"]

        for letter in test_letters:
            self.state.set_group_selection(letter)
            assert self.state.current_group_selection == letter

    def test_non_alphabetic_keys_ignored(self):
        """Test that non-alphabetic keys are ignored."""
        # These should not change the current group selection
        original_group = self.state.current_group_selection
        self.state.set_group_selection("1")  # Number
        self.state.set_group_selection("!")  # Special character
        self.state.set_group_selection("space")  # Word key
        assert self.state.current_group_selection == original_group


class TestFieldProcessing:
    """Test the SourceConfig-based field processing functionality."""

    def test_create_processed_comparison_data(self):
        """Test that field processing works with SourceConfig data."""

        # Define the data
        source_a_data = (
            {"company_name": "Company A", "registration_id": "REG001"},
            {"company_name": "Company B", "registration_id": "REG002"},
            {"company_name": "Company C", "registration_id": "REG003"},
        )

        source_b_data = (
            {"company_name": "Company A Ltd", "address": "123 Main St"},
            {"company_name": "Company B Inc", "address": "456 Oak Ave"},
            {"company_name": "Company C Corp", "address": "789 Pine Rd"},
        )

        data_keys = ["1", "2", "3"]

        # Create sources
        testkit1 = source_from_tuple(
            data_tuple=source_a_data,
            data_keys=data_keys,
            name="source_a",
        )

        testkit2 = source_from_tuple(
            data_tuple=source_b_data,
            data_keys=data_keys,
            name="source_b",
            dag=testkit1.source.dag,
        )

        # Create test DataFrame with qualified field names
        df = pl.DataFrame(
            {
                "leaf": list(range(1, len(source_a_data) + 1)),
                "source_a_company_name": [d["company_name"] for d in source_a_data],
                "source_a_registration_id": [
                    d["registration_id"] for d in source_a_data
                ],
                "source_b_company_name": [d["company_name"] for d in source_b_data],
                "source_b_address": [d["address"] for d in source_b_data],
            }
        )
        # Create evaluation item with new paradigm
        evaluation_item = create_evaluation_item(
            df,
            [
                ("source_a", testkit1.source.config),
                ("source_b", testkit2.source.config),
            ],
            123,
        )

        # Verify structure
        assert isinstance(evaluation_item.display_dataframe, pl.DataFrame)
        assert isinstance(evaluation_item.duplicate_groups, list)
        assert isinstance(evaluation_item.display_columns, list)
        assert isinstance(evaluation_item.leaf_to_display_mapping, dict)
        assert evaluation_item.cluster_id == 123

        # Check expected number of display columns (should be 3 - no duplicates)
        assert len(evaluation_item.display_columns) == 3
        assert len(evaluation_item.duplicate_groups) == 3

        # Check that all leaf IDs are accounted for in duplicate groups
        all_leaf_ids = []
        for group in evaluation_item.duplicate_groups:
            all_leaf_ids.extend(group)
        assert sorted(all_leaf_ids) == [1, 2, 3]

        # Check display dataframe has expected fields
        field_names = evaluation_item.display_dataframe["field_name"].unique().to_list()
        assert "company_name" in field_names
        assert "registration_id" in field_names or "address" in field_names
