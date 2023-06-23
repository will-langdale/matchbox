import duckdb
import pandas as pd
from src.config import stopwords
from src.features.clean_basic import (
    remove_notnumbers_leadingzeroes,
    clean_company_name,
    array_except,
    array_intersect,
    list_join_to_string,
    get_postcode_area,
    get_low_freq_char_sig,
)


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

    return clean.df()


def clean_comp_names(df):
    """
    Lower case, remove punctuation & tokenise the primary company name into an array.
    Extract tokens into: 'unusual' and 'stopwords'. Dedupe. Sort alphabetically.
    Untokenise the unusual words back to a string.
    Args: dataframe containing company_name column
    Returns: dataframe of: company number, 'unusual' tokens', most common 3 tokens,
             most common 4 to 6 tokens, list of previous names of company, postcode.

    """

    # clean and tokenise the two company name fields and put in a new dataframe
    sql_clean_company_name = f"""
    SELECT
        {clean_company_name("company_name")} AS company_name_arr,
        {clean_company_name("secondary_names")} AS secondary_names_arr,
        *
    FROM df
    """
    names_cleaned = duckdb.sql(sql_clean_company_name)  # noqa:F841

    # Define 'stopwords' which are in the company names
    stopword_tokens = pd.DataFrame({"token_array": [stopwords]})  # noqa:F841

    # put the array of stopwords in a separate column alongside the cleaned names
    sql_companies_arr_with_top = """
    select
        *,
        (select * from stopword_tokens) as stopwords
    from names_cleaned
    """
    with_common_terms = duckdb.sql(sql_companies_arr_with_top)  # noqa:F841

    # separate out the company_name into 'unusual' tokens and 'stopwords'
    # TODO: leave name_unusual_tokens (and secondary...) as array & remove split() below
    sql_manipulate_arrays = f"""
    select
        unique_id,
        comp_num_clean,
        {list_join_to_string(
            array_except("company_name_arr", "stopwords")
        )}
            as name_unusual_tokens,
        {list_join_to_string(
            array_except("secondary_names_arr", "stopwords")
        )}
            as secondary_name_unusual_tokens,
        ARRAY_CAT(
            {array_intersect("company_name_arr", "stopwords")},
            {array_intersect("secondary_names_arr", "stopwords")})
            as names_tokens_stopwords,
        postcode,
        null as postcode_alt
    from with_common_terms
    """
    clean = duckdb.sql(sql_manipulate_arrays)

    clean_df = clean.df()

    # finally, in Python because simpler, dedupe names_tokens_stopwords,
    #  sort alphabetically and convert to string
    clean_df["name_unusual_tokens"] = clean_df.name_unusual_tokens.apply(
        lambda x: " ".join(sorted(set(x.split()))) if pd.notnull(x) else x
    )
    clean_df[
        "secondary_name_unusual_tokens"
    ] = clean_df.secondary_name_unusual_tokens.apply(
        lambda x: " ".join(sorted(set(x.split()))) if pd.notnull(x) else x
    )
    # slightly different for the stopwords because already an array so no split() needed
    clean_df["names_tokens_stopwords"] = clean_df.names_tokens_stopwords.apply(
        lambda x: " ".join(set(x))
    )

    # get the first 5 and last 5 chars of name_unusual_tokens, for blocking rules
    clean_df["name_unusual_tokens_first5"] = clean_df.name_unusual_tokens.str[:5]
    clean_df["name_unusual_tokens_last5"] = clean_df.name_unusual_tokens.str[-5:]

    # get first first and last 5 chars of the low frequency character signature

    sql_freq_sig = f"""
        select
            *,
            {get_low_freq_char_sig("name_unusual_tokens")} as name_sig
        from
            clean_df
    """
    clean = duckdb.sql(sql_freq_sig)

    clean_df = clean.df()

    clean_df["name_sig_first5"] = clean_df.name_sig.str[:5]
    clean_df["name_sig_last5"] = clean_df.name_sig.str[-5:]

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

    return clean_df.df()


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
