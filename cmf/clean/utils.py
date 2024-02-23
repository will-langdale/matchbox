from typing import Callable

import duckdb
from pandas import DataFrame

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
    """
    Takes a list of basic cleaning functions appropriate for a select
    statement and add them together into a full cleaning function for use in
    a linker's _clean_data() method. Runs the cleaning in duckdb.

    Only for use with cleaning methods that take a single column as their
    argument. Consider using functools.partial to coerce functions that need
    arguments into this shape, if you want.

    Arguments:
        functions: a list of functions appropriate for a select statement.
        See clean_basic for some examples
    """

    def cleaning_method(df: DataFrame, column: str) -> DataFrame:
        to_run = []

        for f in functions:
            to_run.append(
                f"""
                select
                    *
                    replace ({f(column)} as {column})
                from
                    df;
                """
            )

        for sql in to_run:
            df = duckdb.sql(sql).arrow().to_pandas()

        return df

    return cleaning_method


def alias(function: Callable, alias: str) -> Callable:
    """
    Takes a cleaning function and aliases the output to a new column.

    Arguments:
        function: an outut from a cleaning_function function
    """

    def cleaning_method(df: DataFrame, column: str) -> DataFrame:
        aliased = (
            duckdb.sql(
                f"""
            select
                *,
                {column} as {alias}
            from
                df;
        """
            )
            .arrow()
            .to_pandas()
        )

        processed = function(aliased, alias)

        return processed

    return cleaning_method


def unnest_renest(function: Callable) -> Callable:
    """
    Takes a cleaning function and adds unnesting and renesting either side
    of it. Useful for applying the same function to an array where there are
    sub-functions that also use arrays, blocking list_transform.

    Arguments:
        function: an outut from a cleaning_function function
    """

    def cleaning_method(df: DataFrame, column: str) -> DataFrame:
        unnest = (
            duckdb.sql(
                f"""
        select
            row_number() over () as nest_id,
            *
            replace (unnest({column}) as {column})
        from
            df;
        """
            )
            .arrow()
            .to_pandas()
        )

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

        renest = (
            duckdb.sql(
                f"""
        select
            {any_value_select}
            list({column}) as {column}
        from
            processed
        group by nest_id;
        """
            )
            .arrow()
            .to_pandas()
        )

        return renest

    return cleaning_method
