import duckdb
import pandas as pd
from src.features.clean_basic import (
    remove_notnumbers_leadingzeroes,
    clean_company_name,
    array_except,
    array_intersect,
    list_join_to_string,
    get_postcode_area,
    # get_low_freq_char_sig,
)
from src.config import stopwords


def clean_comp_numbers(df):
    """
    Remove non-numbers, and then leading zeroes
    Args: dataframe containing company_number column
    Returns: dataframe with clean company_number
    """

    sql_clean_comp_number = f"""
    SELECT
        {remove_notnumbers_leadingzeroes("company_number")} as comp_num_clean,
        *
    from df
    """
    clean = duckdb.sql(sql_clean_comp_number)

    return clean


def clean_comp_names(
    df, primary_col: str, secondary_col: str = None, stopwords: str = stopwords
):
    """
    Lower case, remove punctuation & tokenise the primary company name into an array.
    Extract tokens into: 'unusual' and 'stopwords'. Dedupe. Sort alphabetically.
    Untokenise the unusual words back to a string.

    Args:
        df: a dataframe
        primary_col: a column containing the company's main name
        secondary_col: a column containing an array of the company's
            secondary names
        stopwords: a list of stopwords to use for this clean
    Returns:
        dataframe: company number, 'unusual' tokens', most common 3 tokens,
            most common 4 to 6 tokens, list of previous names of company, postcode.
    """

    # TODO: Refactor the silly nested f-strings

    # CLEAN and TOKENISE
    # To a new dataframe
    sql_clean_company_name = f"""
    select
        {clean_company_name(primary_col)} as company_name_arr,
        {
            f"{clean_company_name(secondary_col)} as secondary_names_arr, "
            if secondary_col
            else ""
        }
        *
    from df
    """
    names_cleaned = duckdb.sql(sql_clean_company_name)  # noqa:F841

    # Define STOPWORDS
    # And join them in
    stopword_tokens = pd.DataFrame({"token_array": [stopwords]})  # noqa:F841
    sql_companies_arr_with_top = """
    select
        *,
        (select * from stopword_tokens) as stopwords
    from names_cleaned
    """
    with_common_terms = duckdb.sql(sql_companies_arr_with_top)  # noqa:F841

    # EXTRACT the UNUSUAL and STOPWORD tokens
    # We want the weird stuff from company names
    # TODO: leave name_unusual_tokens (and secondary...) as array & remove split() below
    def secondary_name_unusual_tokens():
        # DuckDB needs a refactor, sorry
        return list_join_to_string(array_except("secondary_names_arr", "stopwords"))

    def cat_names_tokens_stopwords(primary_arr, secondary_arr, stopwords):
        # DuckDB needs a refactor, sorry
        # return array_intersect("secondary_names_arr", "stopwords")
        primary = rf"{array_intersect(primary_arr, stopwords)}"
        secondary = rf"{array_intersect(primary_arr, stopwords)}"

        if secondary_arr:
            return rf"""
                array_cat(
                    {primary},
                    {secondary}
                )
            """
        else:
            return rf"{primary}"

    sql_manipulate_arrays = f"""
    select
        *,
        {
            list_join_to_string(
                array_except("company_name_arr", "stopwords")
            )
        }
            as name_unusual_tokens,
        {
            (
                f"{secondary_name_unusual_tokens()} "
                "as secondary_name_unusual_tokens"
            )
            if secondary_col
            else ""
        }
        {
            cat_names_tokens_stopwords(
                "company_name_arr",
                "secondary_names_arr",
                stopwords
            )
        } as names_tokens_stopwords
    from with_common_terms
    """
    clean = duckdb.sql(sql_manipulate_arrays)

    clean_df = clean.df()

    # DEDUPE names_tokens_stopwords
    clean_df["name_unusual_tokens"] = clean_df.name_unusual_tokens.apply(
        lambda x: " ".join(sorted(set(x.split()))) if pd.notnull(x) else x
    )
    if secondary_col:
        clean_df[
            "secondary_name_unusual_tokens"
        ] = clean_df.secondary_name_unusual_tokens.apply(
            lambda x: " ".join(sorted(set(x.split()))) if pd.notnull(x) else x
        )

    clean_df["names_tokens_stopwords"] = clean_df.names_tokens_stopwords.apply(
        lambda x: " ".join(set(x))
    )

    # Get HEAD and TAIL characters
    # For blocking rules
    clean_df["name_unusual_tokens_first5"] = clean_df.name_unusual_tokens.str[:5]
    clean_df["name_unusual_tokens_last5"] = clean_df.name_unusual_tokens.str[-5:]

    return clean_df


def add_postcode_area(df):
    """
    Extracts postcode area and adds it as a new column.
    Args: dataframe containing postcode column
    Returns: dataframe of: All previous columns plus postcode_area

    """
    sql_add_postcode_area = f"""
    SELECT
        {get_postcode_area("postcode")} AS postcode_area,
        *
    FROM df
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
