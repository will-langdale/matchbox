def characters_to_spaces(input_column):
    """
    Removes all punctuation and replaces with spaces.
    """
    return rf"""
        regexp_replace(
            {input_column},
            '[^a-zA-Z0-9 ]+',
            ' ',
            'g'
        )
    """


def characters_to_nothing(input_column):
    """
    Removes periods and replaces with nothing (U.K. -> UK)
    """

    return rf"""
        regexp_replace(
            {input_column},
            '[.]+',
            '',
            'g'
        )
    """


def clean_punctuation(input_column):
    """
    Set to lower case, remove punctuation
    and replace multiple spaces with single space.
    Finally, trim leading and trailing spaces.
    Args: input_column -- the name of the column to clean
    Returns: string to insert into SQL query
    """

    return rf"""
    trim(
        regexp_replace(
            lower({
                characters_to_spaces(
                    characters_to_nothing(input_column)
                )
            }),
            '\s+',
            ' ',
            'g'
        )
    )
    """


def expand_abbreviations(input_column):
    """
    Expand abbreviations: 'co' to 'company' and 'ltd' to 'limited'.
    Only if followed with a space or at the end of the string.
    Args: input_column -- the name of the column to clean
    Returns: string to insert into SQL query
    """

    return rf"""
    regexp_replace(
        regexp_replace(
            lower({input_column}),
            '(ltd\s|ltd$)',
            'limited ',
            'g'
        ),
        '(co\s|co$)',
        'company ',
        'g'
    )
    """


def tokenise(input_column):
    """
    Split the text in input_column into an array
    using any char that is _not_ alphanumeric, as delimeter
    Args: input_column -- the name of the column to tokenise
    Returns: string to insert into SQL query
    """

    return rf"""
    regexp_split_to_array(
        trim({input_column}),
        '[^a-zA-Z0-9]+'
    )
    """


def dedupe_and_sort(input_column):
    """
    De-duplicate an array of tokens and sort alphabetically
    Args: input_column -- the name of the column to deduplicate (must contain an array)
    Returns: string to insert into SQL query
    """

    return f"""
    array(
        select distinct unnest(
            {input_column}
        ) tokens
        order by tokens
    )
    """


def remove_notnumbers_leadingzeroes(input_column):
    """
    Remove any char that is not a number, then remove all leading zeroes
    Args: input_column -- the name of the column to treat
    Returns: string to insert into SQL query
    """
    return rf"""
    regexp_replace(
        regexp_replace(
            {input_column},
            '[^0-9]',
            '',
            'g'
        ),
        '^0+',
        ''
    )
    """


def clean_company_name_ORIG(input_column):
    """ """
    return f"""
        {
            dedupe_and_sort(
                tokenise(
                expand_abbreviations(
                    clean_punctuation(input_column)
                    )
                )
            )
        }
    """


def clean_company_name(input_column):
    """ """
    return f"""
        {
            tokenise(
                expand_abbreviations(
                    clean_punctuation(input_column)
                    )
            )
        }
    """


def array_except(input_col_name, terms_to_remove):
    return rf"""
    array_filter(
        {input_col_name},
        x -> not array_contains({terms_to_remove}, x)
    )
    """


def array_intersect(input_col_name, terms_to_keep):
    return rf"""
    array_filter(
        {input_col_name},
        x -> array_contains({terms_to_keep}, x)
    )
    """


def regex_remove_list_of_strings(input_col_name, list_of_strings):
    to_remove = "|".join(list_of_strings)
    return rf"""
    trim(
        regexp_replace(
            regexp_replace(
                lower({input_col_name}),
                '{to_remove}',
                '',
                'g'
            ),
        '\s{2,}',
        ' ',
        'g'
        )
    )
    """


def regex_extract_list_of_strings(input_col_name, list_of_strings):
    to_extract = "|".join(list_of_strings)
    return rf"""
    regexp_extract_all({input_col_name}, '{to_extract}', 0)
    """


def list_join_to_string(input_col_name, seperator=" "):
    """ """
    return rf"""list_aggr({input_col_name},
        'string_agg',
        '{seperator}'
    )
    """


def clean_stopwords(input_column):
    return f"""
    {list_join_to_string(dedupe_and_sort(input_column))}
    """


def get_postcode_area(input_column):
    return rf"""
        regexp_extract(
            {input_column},
            '^[a-zA-Z][a-zA-Z]?'
        )
    """


def get_low_freq_char_sig(input_column):
    """
    Removes letters with a frequency of 5% or higher, and spaces
    https://en.wikipedia.org/wiki/Letter_frequency
    """
    return rf"""
        regexp_replace(
            lower({input_column}),
            '[rhsnioate ]+',
            '',
            'g'
        )
    """
