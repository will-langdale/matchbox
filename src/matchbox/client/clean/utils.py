"""Generic utilities for default cleaning functions."""

from typing import Callable

import duckdb
import polars as pl

STOPWORDS = [
    "limited",
    "uk",
    "company",
    "international",
    "group",
    "of",
    "the",
    "inc",
    "and",
    "plc",
    "corporation",
    "llp",
    "pvt",
    "gmbh",
    "u k",
    "pte",
    "usa",
    "bank",
    "b v",
    "bv",
]

ABBREVIATIONS = {"co": "company", "ltd": "limited"}


def cleaning_function(*functions: Callable) -> Callable:
    """Takes a list of basic cleaning functions and composes them into a callable.

    Functions must be appropriate for a select statement.

    Only for use with cleaning methods that take a single column as their argument.
    Consider using functools.partial to coerce functions that need arguments into
    this shape.

    Args:
        functions: a list of functions appropriate for a select statement.
            See clean_basic for some examples
    """

    def cleaning_method(df: pl.DataFrame, column: str) -> pl.DataFrame:
        # Create a single SQL statement that applies all transformations
        # This maintains proper DuckDB context for lambda expressions
        nested_transform = column
        for f in functions:
            nested_transform = f(nested_transform)

        sql = f"""
            select
                *
                replace ({nested_transform} as {column})
            from
                df;
        """

        df_arrow = duckdb.sql(sql).arrow()
        result_df = pl.from_arrow(df_arrow)
        del df_arrow

        return result_df

    return cleaning_method


def alias(function: Callable, alias: str) -> Callable:
    """Takes a cleaning function and aliases the output to a new column.

    Args:
        function: an outut from a cleaning_function function
        alias: the new column name to use
    """

    def cleaning_method(df: pl.DataFrame, column: str) -> pl.DataFrame:
        aliased_sql = f"""
            select
                *,
                {column} as {alias}
            from
                df;
        """
        aliased_arrow = duckdb.sql(aliased_sql).arrow()
        aliased = pl.from_arrow(aliased_arrow)
        del aliased_arrow

        processed = function(aliased, alias)

        return processed

    return cleaning_method


def unnest_renest(function: Callable) -> Callable:
    """Takes a cleaning function and adds unnesting and renesting either side of it.

    Useful for applying the same function to an array where there are sub-functions
    that also use arrays, blocking list_transform.

    Args:
        function: an outut from a cleaning_function function
    """

    def cleaning_method(df: pl.DataFrame, column: str) -> pl.DataFrame:
        unnest_sql = f"""
            select
                row_number() over () as nest_id,
                *
                replace (unnest({column}) as {column})
            from
                df;
        """

        unnest_arrow = duckdb.sql(unnest_sql).arrow()
        unnest = pl.from_arrow(unnest_arrow)
        del unnest_arrow

        processed = function(unnest, column)

        any_value = [
            f"any_value({col}) as {col}"
            for col in processed.columns
            if col not in ["nest_id", column]
        ]
        if len(any_value) > 0:
            any_value_select = f"{', '.join(any_value)}, "
        else:
            any_value_select = ""

        renest_sql = f"""
            select
                {any_value_select}
                list({column}) as {column}
            from
                processed
            group by nest_id;
        """

        renest_arrow = duckdb.sql(renest_sql).arrow()
        renest = pl.from_arrow(renest_arrow)
        del renest_arrow

        return renest

    return cleaning_method
