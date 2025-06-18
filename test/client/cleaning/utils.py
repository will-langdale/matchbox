"""Utilities for programmatic testing of cleaning functions."""

from typing import Any, Callable, Tuple

import duckdb
import polars as pl


def create_test_case(
    input_data: list[Any],
    expected_output: list[Any],
    column_name: str = "col",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Create a test case with dirty and clean dataframes.

    Args:
        input_data: List of dirty values
        expected_output: List of expected clean values
        column_name: Name of the column to use

    Returns:
        A tuple of (dirty, clean) dataframes
    """
    dirty = pl.DataFrame({column_name: input_data})
    clean = pl.DataFrame({column_name: expected_output})

    # Handle list data type inference
    if any(isinstance(item, list) for item in input_data):
        dirty = dirty.with_columns(pl.col(column_name).cast(pl.List(pl.Utf8)))

    if any(isinstance(item, list) for item in expected_output):
        clean = clean.with_columns(pl.col(column_name).cast(pl.List(pl.Utf8)))

    return dirty, clean


def run_cleaner_test(
    cleaner_func: Callable[[str], str],
    input_data: list[Any],
    expected_output: list[Any],
    column_name: str = "col",
) -> Tuple[pl.DataFrame, bool]:
    """Test a cleaner function against input data and expected output.

    Args:
        cleaner_func: The cleaning function to test
        input_data: List of dirty values
        expected_output: List of expected clean values
        column_name: Name of the column to use

    Returns:
        A tuple of (cleaned_df, success)
    """
    dirty, clean = create_test_case(input_data, expected_output, column_name)  # noqa: F841

    # Apply the cleaner function using duckdb
    cleaned = pl.from_arrow(
        duckdb.sql(
            f"""
            select
                {cleaner_func(column_name)} as {column_name}
            from
                dirty
            """
        ).arrow()
    )

    # Handle list-type columns for equality comparison
    if any(isinstance(item, list) for item in expected_output):
        # Convert to Python lists for comparison
        cleaned_values = cleaned[column_name].to_list()
        expected_values = clean[column_name].to_list()

        # Check if lists match by converting to string representation
        cleaned_str = str(cleaned_values)
        expected_str = str(expected_values)

        return cleaned, cleaned_str == expected_str

    return cleaned, cleaned.equals(clean)


def run_composed_test(
    composed_func: Callable[[pl.DataFrame, str], pl.DataFrame],
    input_data: list[Any],
    expected_output: list[Any],
    column_name: str = "col",
) -> Tuple[pl.DataFrame, bool]:
    """Test a composed cleaning function against input data and expected output.

    Args:
        composed_func: The composed cleaning function to test
        input_data: List of dirty values
        expected_output: List of expected clean values
        column_name: Name of the column to use

    Returns:
        A tuple of (cleaned_df, success)
    """
    dirty, clean = create_test_case(input_data, expected_output, column_name)

    # Apply the composed function
    cleaned = composed_func(dirty, column=column_name)

    # Handle list-type columns for equality comparison
    if any(isinstance(item, list) for item in expected_output):
        # Convert to Python lists for comparison
        cleaned_values = cleaned[column_name].to_list()
        expected_values = clean[column_name].to_list()

        # Check if lists match by converting to string representation
        cleaned_str = str(cleaned_values)
        expected_str = str(expected_values)

        return cleaned, cleaned_str == expected_str

    return cleaned, cleaned.equals(clean)
