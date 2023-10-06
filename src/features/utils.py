import duckdb
from typing import Callable


def duckdb_cleaning_factory(functions: list[Callable]) -> Callable:
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
    if not isinstance(functions, list):
        functions = [functions]

    def cleaning_method(df, column: str):
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
            df = duckdb.sql(sql).df()

        return df

    return cleaning_method


def unnest_renest(function: Callable) -> Callable:
    """
    Takes a cleaning function and adds unnesting and renesting either side
    of it. Useful for applying the same function to an array where there are
    sub-functions that also use arrays, blocking list_transform.

    Arguments:
        functions: a cleaning function appropriate for a select statement.
        See clean_basic for some examples
    """

    def cleaning_method(df, column):
        unnest = duckdb.sql(
            f"""
        select
            row_number() over () as nest_id,
            *
            replace (unnest({column}) as {column})
        from
            df;
        """
        ).df()

        processed = function(unnest, column)

        any_value = [
            f"any_value({col}) as {col}"
            for col in processed.columns
            if col not in ["nest_id", column]
        ]

        renest = duckdb.sql(
            f"""
        select
            {", ".join(any_value)},
            list({column}) as {column}
        from
            processed
        group by nest_id;
        """
        ).df()

        return renest

    return cleaning_method
