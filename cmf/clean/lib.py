from functools import partial
from pandas import DataFrame

from cmf.clean import utils as cu
from cmf.clean import steps


def company_name(
    df: DataFrame,
    column: str,
    column_secondary: str = None,
    stopwords: str = cu.STOPWORDS,
) -> DataFrame:
    """
    Lower case, remove punctuation & tokenise the company name into an array.
    Extract tokens into: 'unusual' and 'stopwords'. Dedupe. Sort alphabetically.
    Untokenise the unusual words back to a string.

    Args:
        df: a dataframe
        input_column: a column containing the company's main name
        input_column_secondary: a column containing an array of the company's
            secondary names
        stopwords: a list of stopwords to use for this clean
    Returns:
        dataframe: the same as went in, but cleaned
    """

    remove_stopwords = partial(steps.remove_stopwords, stopwords=stopwords)

    clean_primary = cu.cleaning_function(
        steps.clean_punctuation,
        steps.expand_abbreviations,
        steps.tokenise,  # returns array
        remove_stopwords,
        steps.list_join_to_string,  # returns col
    )

    clean_secondary = cu.unnest_renest(clean_primary)

    df = clean_primary(df, column)

    if column_secondary is not None:
        df = clean_secondary(df, column_secondary)

    return df


def company_number(df: DataFrame, column: str) -> DataFrame:
    """
    Remove non-numbers, and then leading zeroes

    Args:
        df: a dataframe
        input_column: a column containing a company number
    Returns:
        dataframe: the same as went in, but cleaned
    """

    clean_number = cu.cleaning_function(steps.remove_notnumbers_leadingzeroes)

    df = clean_number(df, column)

    return df


def postcode_to_area(df: DataFrame, column: str) -> DataFrame:
    """
    Extracts postcode area from a postcode

    Args:
        df: a dataframe
        input_column: a column containing a postcode
    Returns:
        dataframe: the same as went in, but cleaned
    """

    extract_area = cu.cleaning_function(steps.get_postcode_area)

    df = extract_area(df, column)

    return df
