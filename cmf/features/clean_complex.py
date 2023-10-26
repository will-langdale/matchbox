import duckdb
from functools import partial

from cmf.features.clean_basic import (
    remove_notnumbers_leadingzeroes,
    clean_company_name,
    array_except,
    list_join_to_string,
    get_postcode_area,
    # get_low_freq_char_sig,
)
from cmf.config import stopwords
import cmf.features.utils as fu


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

    clean_primary = fu.duckdb_cleaning_factory(
        [
            clean_company_name,
            remove_stopwords,
            list_join_to_string,
        ]
    )

    clean_secondary = fu.unnest_renest(clean_primary)

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
