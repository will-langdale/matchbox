"""Utility functions for composing SQL cleaning expressions."""

from typing import Callable


def compose(*functions: Callable[[str], str]) -> Callable[[str], str]:
    """Compose SQL string transformations by nesting function calls.

    Functions are applied from left to right, with each function's output
    becoming the input to the next function.

    Useful for composing complex SQL expressions from simpler ones for
    StepInput cleaning.

    Args:
        functions: SQL transformation functions that take a column string
            and return a SQL expression string

    Returns:
        A function that takes a column name and returns a nested SQL expression

    Example:
        ```python
        def to_upper(column: str) -> str:
            return f"UPPER({column})"


        def add_prefix(column: str) -> str:
            return f"'prefix_' || {column}"


        upper_and_prefix = compose(to_upper, add_prefix)

        StepInput(
            prev_node=i_foo,
            select={foo: ["name", "status"]},
            cleaning_dict={
                "name": upper_and_prefix(foo.f("name")),
            },
            combine_type="concat",
        )
        ```
    """

    def _composed_transform(column: str) -> str:
        result = column
        for func in functions:
            result = func(result)
        return result

    return _composed_transform


def vectorise(func: Callable[[str], str]) -> Callable[[str], str]:
    """Vectorise a SQL function to work with array columns using list_transform.

    Takes a function that operates on scalar values and returns a function
    that applies it to each element of an array column using DuckDB's
    list_transform function.

    Args:
        func: A SQL transformation function that takes a column string
            and returns a SQL expression string for scalar operations

    Returns:
        A function that takes an array column name and returns a SQL expression
        that applies the original function to each array element

    Example:
        ```python
        def to_upper(column: str) -> str:
            return f"UPPER({column})"


        upper_array = vectorise(to_upper)

        StepInput(
            prev_node=i_foo,
            select={foo: ["name"]},
            cleaning_dict={
                "name": upper_array(foo.f("name")),
            },
            combine_type="set_agg",
        )
        ```
    """

    def _vectorised_transform(column: str) -> str:
        inner_transform = func("x")
        return f"list_transform({column}, x -> {inner_transform})"

    return _vectorised_transform
