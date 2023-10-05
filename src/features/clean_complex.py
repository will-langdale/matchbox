import duckdb
from typing import Callable
from functools import partial

from src.features.clean_basic import (
    remove_notnumbers_leadingzeroes,
    clean_company_name,
    array_except,
    list_join_to_string,
    get_postcode_area,
    # get_low_freq_char_sig,
)
from src.config import stopwords


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


def clean_comp_names(
    df, primary_col: str, secondary_col: str = None, stopwords: str = stopwords
):
    """
    Lower case, remove punctuation & tokenise the company name into an array.
    Extract tokens into: 'unusual' and 'stopwords'. Dedupe. Sort alphabetically.
    Untokenise the unusual words back to a string.

    Args:
        df: a dataframe
        primary_col: a column containing the company's main name
        secondary_col: a column containing an array of the company's
            secondary names
        stopwords: a list of stopwords to use for this clean
    Returns:
        dataframe: the same as went in, but cleaned
    """

    remove_stopwords = partial(array_except, terms_to_remove=stopwords)

    clean_primary = duckdb_cleaning_factory(
        [
            clean_company_name,
            remove_stopwords,
            list_join_to_string,
        ]
    )

    clean_secondary = unnest_renest(clean_primary)

    df = clean_primary(df, primary_col)

    if secondary_col is not None:
        df = clean_secondary(df, secondary_col)

    return df


def clean_comp_numbers(df):
    """
    Remove non-numbers, and then leading zeroes
    Args: dataframe containing company_number column
    Returns: dataframe with clean company_number
    """

    sql_clean_comp_number = f"""
    select
        {remove_notnumbers_leadingzeroes("company_number")} as comp_num_clean,
        *
    from df
    """
    clean = duckdb.sql(sql_clean_comp_number)

    return clean


def add_postcode_area(df):
    """
    Extracts postcode area and adds it as a new column.
    Args: dataframe containing postcode column
    Returns: dataframe of: All previous columns plus postcode_area

    """
    sql_add_postcode_area = f"""
    select
        {get_postcode_area("postcode")} as postcode_area,
        *
    from df
    """
    clean_df = duckdb.sql(sql_add_postcode_area)  # noqa:F841

    return clean_df


def clean_raw_data(df):
    """
    Steer the cleaning of company numbers and names
    Args: the dataframe to be cleaned
    Returns: the cleaned dataframe
    """

    clean_df = clean_comp_numbers(df)
    clean_df = clean_comp_names(clean_df)
    clean_df = add_postcode_area(clean_df)

    return clean_df
